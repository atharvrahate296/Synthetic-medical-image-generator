import atexit
import html
import os
import random
import re
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread

# This app only ever uses the model trained by train.py. Never touch the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gradio as gr
from PIL import Image
from utils import chat_history, prompt_enhancer
from utils.prompt_enhancer import IMAGE_TYPES


# The fine-tuned model lives in the project folder (written by train.py) and is
# loaded from disk once per app start. Override with MEDSYNTH_MODEL_DIR.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(
    os.environ.get("MEDSYNTH_MODEL_DIR", PROJECT_DIR / "medsynth-model" / "pipeline")
)
_generation_lock = Lock()

IMAGE_TYPES = dict(IMAGE_TYPES)

# Note: no anatomy terms (face, nose, lips, hands...) here on purpose. The
# clinical datasets contain lesions on those body sites.
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, anime, sketch, "
    "3d render, CGI, digital art, artificial skin, plastic skin, "
    "waxy skin, excessively smooth skin, fake texture, "
    "unrealistic pigmentation, neon colors, oversaturated colors, "
    "extreme color grading, dramatic lighting, cinematic lighting, "
    "glowing lesion, excessive contrast, "
    "blur, motion blur, out of focus, low resolution, "
    "compression artifacts, noise, distorted anatomy, "
    "deformed body, duplicated body parts, malformed lesion, "
    "unnatural symmetry, impossible anatomy, "
    "watermark, text, logo, labels, arrows, "
    "UI, border, frame, "
    "food, landscape, scenery"
)

DEFAULT_SETTINGS = dict(chat_history.SETTINGS_DEFAULTS)
DEFAULT_SETTINGS["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
# Clinical prompt enhancement is intentionally always enabled in the UI and generation path.
DEFAULT_SETTINGS["clinical"] = True

TITLE_MAX_CHARS = 30

# Human-readable explanation of the toggle, shown in the settings panel.
ENHANCEMENT_HELP = (
    "The fine-tune was trained on short captions in one fixed grammar - "
    "`<view> of <diagnosis>, <category>, <body site>, <Fitzpatrick type>`. "
    "Enhancement parses what you wrote, normalises it into that grammar, "
    "infers anything you left out, and appends the imaging terms that view "
    "needs, all inside CLIP's hard 75-token budget. Without it your text is "
    "sent verbatim, truncated to the same budget."
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_effective_model_dir() -> Path:
    if (MODEL_DIR / "model_index.json").is_file():
        return MODEL_DIR
    fallback = PROJECT_DIR / "medsynth-model" / "pipeline"
    if (fallback / "model_index.json").is_file():
        return fallback
    fallback_model = PROJECT_DIR / "model" / "pipeline"
    if (fallback_model / "model_index.json").is_file():
        return fallback_model
    return MODEL_DIR


def model_is_available():
    return (get_effective_model_dir() / "model_index.json").is_file()


# The token counter needs the CLIP vocabulary, which lives next to the weights.
_tokenizer_dir = get_effective_model_dir() / "tokenizer"
if _tokenizer_dir.is_dir():
    prompt_enhancer.set_tokenizer_dir(_tokenizer_dir)


@lru_cache(maxsize=1)
def load_pipeline():
    """Load the fine-tuned pipeline from disk (cached for the app's lifetime)."""
    target_dir = get_effective_model_dir()
    if not (target_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"Trained model not found at {target_dir}. Run train.py first "
            "(see README.md), or set MEDSYNTH_MODEL_DIR to the folder that "
            "contains model_index.json."
        )

    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        str(target_dir),
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )

    if device == "cuda":
        try:
            pipeline = pipeline.to(device)
            pipeline.unet.to(memory_format=torch.channels_last)
            pipeline.vae.to(memory_format=torch.channels_last)
        except Exception:  # Catch CUDA OOM or device movement issues on low VRAM GPUs
            print("CUDA memory tight. Enabling attention slicing & CPU offload.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipeline.enable_attention_slicing()
            try:
                pipeline.enable_model_cpu_offload()
            except Exception:
                pipeline = pipeline.to("cpu")
                device = "cpu"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Loaded {target_dir} on CUDA GPU: {torch.cuda.get_device_name(0)} (FP16)")
    else:
        print(f"Loaded {target_dir} on CPU (slow). Install a CUDA build of PyTorch for speed.")

    return pipeline, device


# ---------------------------------------------------------------------------
# Transient image files
# ---------------------------------------------------------------------------

# Images live in the database as BLOBs. Gradio needs a real path to serve them,
# so each stored message is unwrapped once into this scratch directory and the
# path is reused for every later re-render instead of piling up temp files.
_SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="medsynth-chat-"))
_image_paths: dict[int, str] = {}
atexit.register(shutil.rmtree, _SCRATCH_DIR, True)


def _image_file(message_id: int, blob: bytes) -> str:
    cached = _image_paths.get(message_id)
    if cached:
        return cached
    path = _SCRATCH_DIR / f"image-{message_id}.png"
    with open(path, "wb") as handle:
        handle.write(blob)
    _image_paths[message_id] = str(path)
    return str(path)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _short_title(title):
    title = " ".join((title or "Untitled").split())
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title or "Untitled"


def _relative_stamp(value):
    """'2026-09-29 14:27:03' -> 'now' / '14m' / '3h' / '2d' / '12 Oct'."""
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return ""

    delta = datetime.now() - moment
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return moment.strftime("%d %b")
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    if seconds < 86400 * 7:
        return f"{seconds // 86400}d"
    return moment.strftime("%d %b")


def _activity_label(conversation):
    images = conversation.get("image_count") or 0
    messages = conversation.get("message_count") or 0
    if not messages:
        return "empty"
    parts = []
    if images:
        parts.append(f"{images} image" + ("s" if images != 1 else ""))
    parts.append(f"{messages} message" + ("s" if messages != 1 else ""))
    return " · ".join(parts)


def _last_prompt(conversation_id):
    """Most recent user prompt in a thread, used to pre-fill the token meter."""
    for message in reversed(chat_history.get_messages(conversation_id)):
        if message["role"] == "user":
            return str(message["content"] or "")
    return ""


def _conversation_choices(query=""):
    # Two-line labels: the title on the first line, activity + time on the
    # second. The CSS uses `white-space: pre-line` so the newline is honoured.
    # Pinned threads sort to the top and carry a leading dot so the pin is
    # still obvious. Radio labels are plain text, so it must be a glyph.
    choices = []
    for conversation in chat_history.list_conversations(query):
        marker = "● " if conversation["pinned"] else ""
        choices.append(
            (
                f"{marker}{_short_title(conversation['title'])}\n"
                f"{_activity_label(conversation)} · "
                f"{_relative_stamp(conversation['updated_at'])}",
                conversation["id"],
            )
        )
    return choices


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------

PENDING_HTML = """
<div class="ms-pending" role="status" aria-live="polite">
  <div class="ms-pending__frame"><div class="ms-pending__scan"></div></div>
  <div class="ms-pending__body">
    <span class="ms-pending__label">Generating image</span>
    <span class="ms-pending__hint">Diffusion sampling - this can take a
      moment on a laptop GPU</span>
  </div>
</div>
"""


def _error_html(message):
    return (
        '<div class="ms-error" role="alert">'
        '<span class="ms-error__icon" aria-hidden="true">!</span>'
        f'<span class="ms-error__text">{html.escape(str(message))}</span>'
        "</div>"
    )


def _render_conversation(conversation_id, pending=False):
    rendered = []

    for message in chat_history.get_messages(conversation_id):
        content = str(message["content"] or "")
        is_error = content.startswith("Generation failed:")

        rendered.append(
            {
                "role": message["role"],
                "content": _error_html(content) if is_error else content,
            }
        )

        if message["image"] is None:
            continue

        try:
            # Re-validate through PIL so a truncated BLOB cannot poison Gradio.
            image = Image.open(BytesIO(message["image"])).convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            path = _image_file(message["id"], buffer.getvalue())
            rendered.append(
                {
                    "role": "assistant",
                    "content": {
                        "path": path,
                        "alt_text": _plain_text(content)
                        or "Generated synthetic skin image",
                    },
                }
            )
        except Exception as error:
            rendered.append(
                {
                    "role": "assistant",
                    "content": _error_html(f"Unable to display image: {error}"),
                }
            )

    if pending:
        rendered.append({"role": "assistant", "content": PENDING_HTML})

    return rendered


# ---------------------------------------------------------------------------
# Sidebar state
# ---------------------------------------------------------------------------

def _history_update(selected_id=None, query=""):
    choices = _conversation_choices(query)
    conversation_ids = {value for _, value in choices}
    if selected_id not in conversation_ids:
        selected_id = choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected_id), selected_id


def _history_empty_html(query, total):
    if total == 0:
        return (
            '<div class="ms-empty">'
            "<strong>No conversations yet</strong>"
            "<span>Start one with New conversation, then describe a lesion."
            "</span></div>"
        )
    if query.strip():
        return (
            '<div class="ms-empty">'
            f"<strong>No match for “{html.escape(query.strip())}”</strong>"
            "<span>Search looks at titles and everything said inside.</span>"
            "</div>"
        )
    return ""


def _history_sidebar(selected_id=None, query=""):
    update, selected_id = _history_update(selected_id, query)
    total = len(chat_history.list_conversations())
    return update, _history_empty_html(query, total), selected_id


def _settings_updates(settings):
    return (
        settings["image_type"],
        True,
        int(settings["steps"]),
        float(settings["guidance"]),
        int(settings["seed"]),
        settings["negative_prompt"] or DEFAULT_NEGATIVE_PROMPT,
    )


# ---------------------------------------------------------------------------
# Prompt context bar
# ---------------------------------------------------------------------------

def _chip(label, value, tone=""):
    tone = f" ms-chip--{tone}" if tone else ""
    return (
        f'<span class="ms-chip{tone}">'
        f'<span class="ms-chip__key">{html.escape(label)}</span>'
        f'<span class="ms-chip__val">{html.escape(str(value))}</span>'
        "</span>"
    )


def _context_html(prompt, image_type, clinical, steps, guidance, seed):
    """Compact prompt-cost indicator shown above the transcript."""
    plan = prompt_enhancer.build_plan(prompt, True, image_type)
    fill = max(0, min(100, round(plan.usage * 100)))
    tone = "warn" if plan.truncated or plan.tokens >= plan.limit - 5 else "normal"
    return (
        f'<div class="ms-tokenbar ms-tokenbar--{tone}" role="status" '
        f'aria-label="Prompt token usage: {plan.tokens} of {plan.limit} tokens">'
        f'<span class="ms-tokenbar__label">Prompt cost</span>'
        f'<span class="ms-tokenbar__track"><span class="ms-tokenbar__fill" style="width:{fill}%"></span></span>'
        f'<strong>{plan.tokens}/{plan.limit}</strong>'
        f'<span class="ms-tokenbar__unit">tokens</span>'
        f'</div>'
    )


def sync_context(prompt, image_type, clinical, steps, guidance, seed):
    return _context_html(prompt, image_type, clinical, steps, guidance, seed)


# ---------------------------------------------------------------------------
# Settings actions
# ---------------------------------------------------------------------------

def persist_settings(
    conversation_id, image_type, clinical, steps, guidance, seed,
    negative_prompt, prompt,
):
    """Store the current controls on this conversation, and echo the context bar."""
    if conversation_id:
        chat_history.save_settings(
            conversation_id,
            image_type=image_type,
            clinical=True,
            steps=int(steps),
            guidance=float(guidance),
            seed=-1 if seed is None else int(seed),
            negative_prompt=negative_prompt or "",
        )
    return sync_context(prompt, image_type, clinical, steps, guidance, seed)


def reset_settings(conversation_id, prompt=""):
    settings = chat_history.reset_settings(conversation_id)
    settings["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
    updates = _settings_updates(settings)
    return (*updates, sync_context(prompt, *updates[:5]))


def randomize_seed():
    return random.randint(0, 2_147_483_647)


# ---------------------------------------------------------------------------
# Conversation actions
# ---------------------------------------------------------------------------
#
# Every action that can change which conversation is open returns the same
# tuple, in the order of CONVERSATION_OUTPUTS (declared with the UI below).
# Routing all of them through one refresh function means the sidebar, the
# transcript, the settings and the meta bar can never drift out of sync with
# each other, and adding an output only means editing one place.

UNARMED = gr.update(value="Delete conversation", elem_classes=[])


def _refresh(conversation_id, query="", *, transcript=None, prompt="",
             search=""):
    """The canonical update for `conversation_id`, ready to be returned."""
    history_update, empty_html, selected_id = _history_sidebar(
        conversation_id, query,
    )
    updates = _settings_updates(chat_history.get_settings(selected_id))
    conversation = chat_history.get_conversation(selected_id) or {}
    pinned = bool(conversation.get("pinned"))

    return (
        _render_conversation(selected_id) if transcript is None else transcript,
        selected_id,
        history_update,
        empty_html,
        *updates,
        sync_context(prompt or _last_prompt(selected_id), *updates[:5]),
        UNARMED,
        None,
        search,
        gr.update(value="Unpin" if pinned else "Pin",
                  elem_classes=["ms-btn--on"] if pinned else []),
        conversation.get("title") or "",
        "",
    )


def start_conversation(query=""):
    """Create a thread and clear the search filter so it is actually visible.

    Without clearing it, a new thread would fall outside the active filter and
    the sidebar would highlight some other conversation than the one opened.
    """
    conversation_id = chat_history.create_conversation(
        {
            "image_type": DEFAULT_SETTINGS["image_type"],
            "clinical": True,
            "steps": DEFAULT_SETTINGS["steps"],
            "guidance": DEFAULT_SETTINGS["guidance"],
            "seed": DEFAULT_SETTINGS["seed"],
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        }
    )
    # Several abandoned clicks in a row should not leave a column of empty
    # threads behind; the newest one always survives.
    chat_history.prune_empty_conversations()
    return _refresh(conversation_id, query="", transcript=[], prompt="")


def select_conversation(conversation_id, query=""):
    return _refresh(conversation_id, query)


def remove_conversation(conversation_id, query=""):
    chat_history.delete_conversation(conversation_id)
    if not chat_history.list_conversations():
        chat_history.create_conversation()
    return _refresh(None, query)


def clear_conversation(conversation_id, query=""):
    """Empty the thread but keep it - and its settings - in the sidebar."""
    chat_history.clear_messages(conversation_id)
    chat_history.rename_conversation(conversation_id, "New conversation")
    return _refresh(conversation_id, query, transcript=[])


def filter_history(query, conversation_id):
    history_update, empty_html, selected_id = _history_sidebar(
        conversation_id, query,
    )
    return history_update, empty_html, selected_id


def rename_conversation_from_ui(conversation_id, title):
    """Save a user-supplied name, or clear the field when it was left empty."""
    conversation_id = conversation_id or chat_history.create_conversation()
    title = (title or "").strip()
    if title:
        chat_history.rename_conversation(conversation_id, title)
    else:
        conversation = chat_history.get_conversation(conversation_id)
        title = (conversation or {}).get("title") or ""

    update, empty_html, selected_id = _history_sidebar(conversation_id)
    return update, empty_html, gr.update(value=title)


def toggle_pin(conversation_id, query=""):
    """Pin or unpin, then keep the thread selected and in view."""
    chat_history.set_pinned(conversation_id)
    return _refresh(conversation_id, query, transcript=gr.skip())


def duplicate_conversation(conversation_id, query=""):
    """Branch a copy of this thread and open it."""
    new_id = chat_history.duplicate_conversation(conversation_id)
    if new_id is None:
        return _refresh(conversation_id, query, transcript=gr.skip())
    return _refresh(new_id, query="")


def export_conversation(conversation_id):
    """Write the thread to results/exports and say where it landed."""
    path = chat_history.export_conversation(conversation_id)
    if path is None:
        return (
            '<div class="ms-note-box ms-note-box--error">Nothing to export.'
            "</div>"
        )
    folder = path.parent
    return (
        f'<div class="ms-note-box ms-note-box--ok">'
        f"Exported to <code>{html.escape(str(folder))}</code>"
        "</div>"
    )


def arm_delete(conversation_id, pending_id):
    """First click arms, second click deletes. Never a one-click data loss.

    Gradio requires one return value per declared output, so the arming branch
    pads with `gr.skip()` to leave the transcript and settings untouched.
    """
    skip = [gr.skip()] * 16

    if not conversation_id:
        return (
            *skip,
            gr.update(value="Nothing to delete", elem_classes=["ms-btn--armed"]),
            None,
        )

    if pending_id != conversation_id:
        return (
            *skip,
            gr.update(value="Confirm delete", elem_classes=["ms-btn--armed"]),
            conversation_id,
        )

    return (*remove_conversation(conversation_id), None)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _metadata_parts(plan, image_type, steps, guidance, used_seed):
    """The one-line summary stored with an image: (caption, facts)."""
    caption = plan.as_caption() if plan.enhanced else (plan.text or "custom prompt")
    facts = [
        f"{int(steps)} steps",
        f"CFG {float(guidance):g}",
        f"seed {used_seed}",
        f"{plan.tokens} tokens",
    ]
    if not plan.enhanced:
        facts.insert(0, "verbatim")
    elif plan.truncated:
        facts.insert(0, "trimmed")
    return caption, facts


def _metadata_caption(plan, image_type, steps, guidance, used_seed):
    caption, facts = _metadata_parts(plan, image_type, steps, guidance, used_seed)
    return (
        '<div class="ms-cap">'
        f'<b>{html.escape(image_type)}</b> · {html.escape(caption)}<br>'
        f'<span class="ms-cap__meta">{html.escape(" · ".join(facts))}</span>'
        "</div>"
    )


def _plain_text(markup):
    """Flatten stored caption markup for screen readers and alt text."""
    text = re.sub(r"<br\s*/?>", ". ", str(markup or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).replace(" · ", ", ").replace("..", ".").strip()


def generate_image(
    prompt,
    conversation_id,
    image_type,
    steps,
    guidance_scale,
    seed,
    medical_prompt_enabled,
    negative_prompt,
):
    prompt = (prompt or "").strip()
    # Clinical prompt enhancement is a core part of the Medsynth generation path.
    medical_prompt_enabled = True

    plan = prompt_enhancer.build_plan(
        prompt, True, image_type,
    )

    if not prompt:
        yield (
            _render_conversation(conversation_id),
            "",
            conversation_id,
            gr.update(),
            gr.update(),
            gr.update(),
            sync_context("", image_type, medical_prompt_enabled,
                        steps, guidance_scale, seed),
        )
        return

    if not conversation_id:
        conversation_id = chat_history.create_conversation()

    chat_history.add_message(conversation_id, "user", prompt)
    chat_history.save_settings(
        conversation_id,
        image_type=image_type,
        clinical=bool(medical_prompt_enabled),
        steps=int(steps),
        guidance=float(guidance_scale),
        seed=-1 if seed is None else int(seed),
        negative_prompt=negative_prompt or "",
    )

    history_update, empty_html, _ = _history_sidebar(conversation_id)
    yield (
        _render_conversation(conversation_id, pending=True),
        "",
        conversation_id,
        history_update,
        empty_html,
        gr.update(),
        sync_context(prompt, image_type, medical_prompt_enabled,
                     steps, guidance_scale, seed),
    )

    used_seed = -1 if seed is None else int(seed)

    try:
        import torch

        with _generation_lock:
            pipeline, device = load_pipeline()

            generator = None
            if used_seed >= 0:
                generator = torch.Generator(device=device).manual_seed(used_seed)
            else:
                # Report the seed that was actually used so a good result can
                # be reproduced by pasting it back into the seed field.
                used_seed = random.randint(0, 2_147_483_647)
                generator = torch.Generator(device=device).manual_seed(used_seed)

            negative = (negative_prompt or "").strip() or None
            image = pipeline(
                prompt=plan.text,
                negative_prompt=negative,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
            ).images[0]

        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        chat_history.add_message(
            conversation_id,
            "assistant",
            _metadata_caption(
                plan, image_type, steps, guidance_scale, used_seed,
            ),
            image=image_buffer.getvalue(),
            image_mime="image/png",
        )
    except Exception as error:
        chat_history.add_message(
            conversation_id,
            "assistant",
            f"Generation failed: {error}",
        )

    history_update, empty_html, _ = _history_sidebar(conversation_id)
    yield (
        _render_conversation(conversation_id),
        "",
        conversation_id,
        history_update,
        empty_html,
        gr.update(),
        sync_context("", image_type, medical_prompt_enabled,
                     steps, guidance_scale, seed),
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

initial_choices = _conversation_choices()
if not initial_choices:
    chat_history.create_conversation()
    initial_choices = _conversation_choices()
initial_conversation_id = initial_choices[0][1]
initial_settings = chat_history.get_settings(initial_conversation_id)
initial_settings["clinical"] = True
if not initial_settings["negative_prompt"]:
    initial_settings["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT
initial_updates = _settings_updates(initial_settings)
initial_conversation = chat_history.get_conversation(initial_conversation_id) or {}
initial_title = initial_conversation.get("title") or ""
initial_pinned = bool(initial_conversation.get("pinned"))


_model_state: dict[str, str] = {"state": "loading", "detail": ""}


def _status_pill(tone, label):
    return (
        f'<div class="ms-status ms-status--{tone}" role="status" aria-live="polite">'
        '<span class="ms-status__dot" aria-hidden="true"></span>'
        f"{label}</div>"
    )


def model_status_html():
    """A small pill in the sidebar so the model state is never a mystery."""
    state, detail = _model_state["state"], _model_state["detail"]

    if not model_is_available():
        return _status_pill(
            "error", "No trained model &mdash; run <code>train.py</code> first",
        )
    if state == "error":
        return _status_pill("error", f"Model failed to load &mdash; {detail}")
    if state == "loading":
        return _status_pill("loading", "Preparing local model&hellip;")
    return _status_pill("ready", f"Model ready &middot; {detail}")


def model_status_tick():
    """Keep the sidebar status synchronized with the actual warm-up state."""
    return model_status_html()


def _describe_device():
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
            return f"{name} &middot; FP16"
        return "CPU &middot; slow"
    except Exception:
        return "device unknown"


# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------

HEAD_HTML = r"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap">
<script>
(function () {
  "use strict";

  const VIEW = "#conversation-view";
  const PROMPT = "#prompt-box textarea";
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  function transcriptScroller() {
    return document.querySelector(VIEW + " .wrap");
  }

  function autosize(textarea) {
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
  }

  function stickToBottom(behavior) {
    const view = transcriptScroller();
    if (!view) return;
    view.scrollTo({
      top: view.scrollHeight,
      behavior: behavior || (reduced.matches ? "auto" : "smooth")
    });
  }

  function syncJumpButton() {
    const view = transcriptScroller();
    const button = document.querySelector("#scroll-controls .ms-jump");
    if (!view || !button) return;
    const distance = view.scrollHeight - view.scrollTop - view.clientHeight;
    const show = view.scrollHeight > view.clientHeight + 40 && distance > 100;
    button.hidden = !show;
    button.classList.toggle("is-visible", show);
  }

  function openLightbox(source) {
    const overlay = document.createElement("div");
    overlay.className = "ms-lightbox";
    overlay.innerHTML =
      '<div class="ms-lightbox__panel">' +
      '<button class="ms-lightbox__close" type="button" aria-label="Close">&times;</button>' +
      '<img alt="Generated image">' +
      '<div class="ms-lightbox__caption"></div>' +
      '</div>';

    const image = overlay.querySelector("img");
    image.src = source.currentSrc || source.src;
    image.alt = source.alt || "Generated image";

    const row = source.closest(".message-row");
    const caption = row && row.querySelector(".ms-cap");
    overlay.querySelector(".ms-lightbox__caption").textContent =
      caption ? caption.textContent.replace(/\s+/g, " ").trim() : image.alt;

    const close = () => {
      overlay.remove();
      document.removeEventListener("keydown", onKey);
    };
    const onKey = (event) => {
      if (event.key === "Escape") close();
    };

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay || event.target.classList.contains("ms-lightbox__panel")) close();
    });
    overlay.querySelector(".ms-lightbox__close").addEventListener("click", close);
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add("is-open"));
  }

  function decorateMessages() {
    document.querySelectorAll(VIEW + " .message-row").forEach((row) => {
      const message = row.querySelector(".message");
      if (!message || row.querySelector(".ms-copy")) return;

      const button = document.createElement("button");
      button.type = "button";
      button.className = "ms-copy";
      button.textContent = "Copy";
      button.setAttribute("aria-label", "Copy message text");
      button.addEventListener("click", async () => {
        const text = (message.innerText || "").trim();
        try {
          await navigator.clipboard.writeText(text);
        } catch (_) {
          const area = document.createElement("textarea");
          area.value = text;
          document.body.appendChild(area);
          area.select();
          document.execCommand("copy");
          area.remove();
        }
        button.textContent = "Copied";
        button.classList.add("is-done");
        setTimeout(() => {
          button.textContent = "Copy";
          button.classList.remove("is-done");
        }, 1400);
      });
      row.appendChild(button);
    });

    document.querySelectorAll(VIEW + " img").forEach((image) => {
      if (image.dataset.msZoom === "on") return;
      image.dataset.msZoom = "on";
      image.classList.add("ms-zoomable");
      image.addEventListener("click", () => openLightbox(image));
    });
  }


  function toggleMobileSidebar() {
    const sidebar = document.querySelector("#app-sidebar");
    const backdrop = document.querySelector("#mobile-backdrop");
    if (!sidebar || !backdrop) return;
    const open = !sidebar.classList.contains("is-open");
    sidebar.classList.toggle("is-open", open);
    backdrop.classList.toggle("is-visible", open);
    document.body.classList.toggle("ms-sidebar-open", open);
  }

  function closeMobileSidebar() {
    const sidebar = document.querySelector("#app-sidebar");
    const backdrop = document.querySelector("#mobile-backdrop");
    if (!sidebar || !backdrop) return;
    sidebar.classList.remove("is-open");
    backdrop.classList.remove("is-visible");
    document.body.classList.remove("ms-sidebar-open");
  }

  function focusPrompt() {
    const prompt = document.querySelector(PROMPT);
    if (!prompt) return;
    prompt.focus({ preventScroll: true });
  }

  function isTyping(target) {
    if (!target) return false;
    return ["TEXTAREA", "INPUT", "SELECT"].includes(target.tagName) || target.isContentEditable;
  }

  function onKeydown(event) {
    if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
    const key = event.key.toLowerCase();

    if (key === "enter") {
      const prompt = document.querySelector(PROMPT);
      if (prompt && prompt.value.trim()) {
        event.preventDefault();
        prompt.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      }
      return;
    }

    if (isTyping(event.target)) return;
    if (key === "k") {
      event.preventDefault();
      const button = document.querySelector("#new-chat button");
      if (button) button.click();
    } else if (key === "/") {
      event.preventDefault();
      focusPrompt();
    } else if (key === "f") {
      event.preventDefault();
      const search = document.querySelector("#history-search input");
      if (search) search.focus();
    }
  }

  function boot() {
    const prompt = document.querySelector(PROMPT);
    if (prompt && !prompt.dataset.msBound) {
      prompt.dataset.msBound = "1";
      autosize(prompt);
      prompt.addEventListener("input", () => autosize(prompt));
      prompt.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          if (prompt.form && prompt.form.requestSubmit) prompt.form.requestSubmit();
          else prompt.blur();
        }
      });
    }

    const view = transcriptScroller();
    if (view && !view.dataset.msBound) {
      view.dataset.msBound = "1";
      view.addEventListener("scroll", syncJumpButton, { passive: true });
    }

    const jump = document.querySelector("#scroll-controls .ms-jump");
    if (jump && !jump.dataset.msBound) {
      jump.dataset.msBound = "1";
      jump.addEventListener("click", () => stickToBottom());
    }

    const mobileToggle = document.querySelector("#mobile-menu button");
    if (mobileToggle && !mobileToggle.dataset.msBound) {
      mobileToggle.dataset.msBound = "1";
      mobileToggle.addEventListener("click", toggleMobileSidebar);
    }

    const backdrop = document.querySelector("#mobile-backdrop");
    if (backdrop && !backdrop.dataset.msBound) {
      backdrop.dataset.msBound = "1";
      backdrop.addEventListener("click", closeMobileSidebar);
    }

    document.querySelectorAll("#app-sidebar button, #history-list label").forEach((node) => {
      if (node.dataset.msCloseBound) return;
      node.dataset.msCloseBound = "1";
      node.addEventListener("click", () => {
        if (window.matchMedia("(max-width: 900px)").matches) setTimeout(closeMobileSidebar, 80);
      });
    });

    document.querySelectorAll("#conversation-view .placeholder-content code").forEach((chip) => {
      if (chip.dataset.msSuggestionBound) return;
      chip.dataset.msSuggestionBound = "1";
      chip.setAttribute("role", "button");
      chip.setAttribute("tabindex", "0");
      chip.addEventListener("click", () => {
        const text = chip.textContent.replace(/^[`"']|[`"']$/g, "").trim();
        const textarea = document.querySelector(PROMPT);
        if (textarea && text) {
          textarea.value = text;
          autosize(textarea);
          textarea.dispatchEvent(new Event("input", { bubbles: true }));
          textarea.focus();
        }
      });
    });

    decorateMessages();
    syncJumpButton();
    syncPromptDetails();
    if (view) stickToBottom("auto");
  }

  const observeTarget = document.querySelector("#conversation-view");
  if (observeTarget) {
    new MutationObserver(() => {
      decorateMessages();
      syncJumpButton();
      boot();
    }).observe(observeTarget, { childList: true, subtree: true });
  }

  document.addEventListener("keydown", onKeydown);
  document.addEventListener("DOMContentLoaded", boot);
  if (document.readyState !== "loading") boot();
})();
</script>
"""

APP_CSS = """
/* ========================================================================
   Medsynth UI — restrained clinical workspace
   ======================================================================== */
.gradio-container.gradio-container {
    --font: "Instrument Sans", ui-sans-serif, system-ui, sans-serif;
    --display: "Bricolage Grotesque", "Instrument Sans", ui-sans-serif, system-ui, sans-serif;
    --bg: #eef3f2;
    --sidebar: #f8fbfa;
    --surface: #ffffff;
    --surface-2: #f0f6f4;
    --surface-3: #e3efec;
    --line: #d4e1dd;
    --line-strong: #aebfb9;
    --text: #10231f;
    --muted: #64746f;
    --accent: #087f73;
    --accent-strong: #075f58;
    --accent-rgb: 8 127 115;
    --on-accent: #ffffff;
    --danger: #b42318;
    --danger-rgb: 180 35 24;
    --warn: #a16207;
    --shadow: 0 12px 30px -22px rgb(16 32 28 / .45);
    --shadow-lg: 0 20px 55px -30px rgb(16 32 28 / .42);
    --radius-sm: 9px;
    --radius-md: 13px;
    --radius-lg: 18px;
    --radius-xl: 22px;
    --sidebar-width: clamp(280px, 22vw, 340px);
    --ease: cubic-bezier(.2,.7,.2,1);

    --body-background-fill: var(--bg);
    --body-text-color: var(--text);
    --body-text-color-subdued: var(--muted);
    --background-fill-primary: var(--surface);
    --background-fill-secondary: var(--surface-2);
    --border-color-primary: var(--line);
    --border-color-accent: rgb(var(--accent-rgb) / .45);
    --border-color-accent-subdued: rgb(var(--accent-rgb) / .22);
    --color-accent: var(--accent);
    --color-accent-soft: rgb(var(--accent-rgb) / .10);
    --block-background-fill: var(--surface);
    --block-border-color: var(--line);
    --block-border-width: 1px;
    --block-radius: var(--radius-md);
    --block-shadow: none;
    --input-background-fill: var(--surface-2);
    --input-background-fill-hover: var(--surface-2);
    --input-background-fill-focus: var(--surface);
    --input-border-color: var(--line);
    --input-border-color-hover: var(--line-strong);
    --input-border-color-focus: var(--accent);
    --input-radius: var(--radius-sm);
    --input-shadow: none;
    --input-shadow-focus: 0 0 0 3px rgb(var(--accent-rgb) / .16);
    --button-primary-background-fill: var(--accent);
    --button-primary-background-fill-hover: var(--accent-strong);
    --button-primary-text-color: var(--on-accent);
    --button-primary-border-color: transparent;
    --button-secondary-background-fill: var(--surface);
    --button-secondary-background-fill-hover: var(--surface-2);
    --button-secondary-text-color: var(--text);
    --button-secondary-border-color: var(--line);
    --checkbox-background-color-selected: var(--accent);
    --slider-color: var(--accent);
    --chatbot-text-size: 14.5px;
}

.dark .gradio-container.gradio-container,
.gradio-container.gradio-container.dark {
    --bg: #081012;
    --sidebar: #091518;
    --surface: #101d21;
    --surface-2: #15262b;
    --surface-3: #1b3036;
    --line: #20353b;
    --line-strong: #2d4850;
    --text: #e8f3f0;
    --muted: #8ea6a0;
    --accent: #41e0c4;
    --accent-strong: #6ef2da;
    --accent-rgb: 65 224 196;
    --on-accent: #06211e;
    --danger: #f87171;
    --danger-rgb: 248 113 113;
    --warn: #fbbf24;
    --shadow: 0 12px 32px -20px rgb(0 0 0 / .7);
    --shadow-lg: 0 28px 70px -30px rgb(0 0 0 / .8);
}

html, body {
    margin: 0 !important;
    width: 100%;
    height: 100dvh;
    overflow: hidden !important;
    background: var(--bg);
}
body { font-family: var(--font); color: var(--text); }
body.ms-sidebar-open { overflow: hidden !important; }
gradio-app, .gradio-container { width: 100%; height: 100%; }
.gradio-container.gradio-container {
    max-width: none !important;
    height: 100dvh !important;
    min-height: 100dvh !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    background: var(--bg);
    color: var(--text);
}
.gradio-container main { width: 100% !important; height: 100% !important; max-width: none !important; padding: 0 !important; }
.gradio-container footer { display: none !important; }

/* ----------------------------- shell ----------------------------------- */
#app-shell {
    display: flex !important;
    flex-wrap: nowrap !important;
    width: 100% !important;
    height: 100dvh !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
    overflow: hidden !important;
}
#app-shell > * { min-width: 0 !important; box-sizing: border-box; }
#app-shell .form { border: 0 !important; background: transparent !important; box-shadow: none !important; }

/* ----------------------------- sidebar --------------------------------- */
#app-sidebar {
    position: relative;
    display: flex !important;
    flex-direction: column !important;
    flex: 0 0 var(--sidebar-width) !important;
    width: var(--sidebar-width) !important;
    max-width: var(--sidebar-width) !important;
    height: 100dvh !important;
    padding: 20px 14px 14px !important;
    gap: 10px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    scrollbar-width: thin;
    background: var(--sidebar) !important;
    border-right: 1px solid var(--line) !important;
    z-index: 30;
}
#brand, #chat-heading { border: 0 !important; background: transparent !important; box-shadow: none !important; }
#brand { padding: 0 6px !important; }
#brand h1 {
    display: flex; align-items: center; gap: 9px;
    margin: 0 0 2px !important;
    font: 700 21px/1.1 var(--display) !important;
    letter-spacing: -.025em;
}
#brand h1::before {
    content: ""; width: 25px; height: 25px; flex: 0 0 25px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--accent) 0 25%, transparent 28% 50%, rgb(var(--accent-rgb) / .25) 52% 100%);
    box-shadow: 0 0 12px rgb(var(--accent-rgb) / .35);
}
#brand p { margin: 0 !important; font-size: 11.5px; color: var(--muted); }

#model-status { margin: 0 2px 2px; }
.ms-status {
    display: flex; align-items: center; gap: 7px;
    min-height: 30px; box-sizing: border-box;
    padding: 6px 10px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--surface);
    color: var(--muted); font-size: 11px; line-height: 1.25;
}
.ms-status__dot { width: 7px; height: 7px; flex: 0 0 7px; border-radius: 50%; background: var(--line-strong); }
.ms-status--ready .ms-status__dot { background: var(--accent); box-shadow: 0 0 7px var(--accent); }
.ms-status--loading .ms-status__dot { background: var(--warn); box-shadow: 0 0 8px rgb(161 98 7 / .35); animation: ms-pulse 1.2s ease-in-out infinite; }
.ms-status--error { border-color: rgb(var(--danger-rgb) / .35); color: var(--danger); background: rgb(var(--danger-rgb) / .06); }
.ms-status--error .ms-status__dot { background: var(--danger); }

#new-chat {
    width: 100% !important; min-height: 42px !important;
    border-radius: var(--radius-md) !important;
    font-size: 13px !important; font-weight: 600 !important;
    box-shadow: 0 8px 18px -12px rgb(var(--accent-rgb) / .8) !important;
    transition: transform .18s var(--ease), box-shadow .18s var(--ease) !important;
}
#new-chat:hover { transform: translateY(-1px); box-shadow: 0 12px 22px -12px rgb(var(--accent-rgb) / .9) !important; }

.ms-label {
    margin: 0 0 5px 2px !important;
    font: 700 10px/1.1 var(--font) !important;
    text-transform: uppercase; letter-spacing: .09em;
    color: var(--muted) !important;
}
#history-form { display: flex !important; flex-direction: column !important; flex: 1 1 280px !important; min-height: 190px !important; gap: 7px !important; overflow: hidden !important; }
#history-search { margin: 0 !important; }
#history-search input {
    height: 36px !important; box-sizing: border-box;
    padding: 8px 11px !important;
    border-radius: var(--radius-sm) !important;
    background: var(--surface) !important;
    font-size: 12px !important;
}
#history-list {
    flex: 1 1 auto !important; min-height: 0 !important;
    overflow-y: auto !important; overflow-x: hidden !important;
    border: 0 !important; background: transparent !important; padding: 0 !important;
}
#history-list .wrap, #history-list .options, #history-list div[role="radiogroup"], #history-list .radio-group {
    display: flex !important; flex-direction: column !important; gap: 5px !important;
    min-height: min-content !important; overflow: visible !important;
}
#history-list label {
    position: relative !important;
    display: flex !important; align-items: center !important;
    width: 100% !important; min-height: 52px !important; box-sizing: border-box !important;
    margin: 0 !important; padding: 8px 11px 8px 13px !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-sm) !important;
    background: transparent !important;
    cursor: pointer !important;
    transition: background .16s var(--ease), border-color .16s var(--ease), transform .16s var(--ease) !important;
}
#history-list label:hover { background: var(--surface); border-color: var(--line); transform: translateX(2px); }
#history-list label.selected { background: rgb(var(--accent-rgb) / .09) !important; border-color: rgb(var(--accent-rgb) / .32) !important; }
#history-list label.selected::before { content: ""; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px; border-radius: 3px; background: var(--accent); }
#history-list label span { width: 100% !important; font-size: 11px !important; line-height: 1.35 !important; color: var(--muted) !important; white-space: pre-line !important; overflow: hidden !important; text-overflow: ellipsis !important; }
#history-list label span::first-line { font-size: 12.5px !important; font-weight: 600 !important; color: var(--text) !important; }
#history-list label.selected span::first-line { color: var(--accent) !important; }
#history-list input[type="radio"] { position: absolute; opacity: 0; pointer-events: none; }

#active-thread-panel {
    flex: 0 0 auto !important; gap: 7px !important;
    padding: 11px !important; margin: 0 !important;
    border: 1px solid var(--line) !important; border-radius: var(--radius-md) !important;
    background: var(--surface) !important; box-shadow: var(--shadow) !important;
}
#rename-title input { height: 34px !important; font-size: 12px !important; }
#manage-row-1, #manage-row-2 { gap: 6px !important; margin: 0 !important; }
#manage-row-1 button, #manage-row-2 button { min-height: 33px !important; padding: 5px 7px !important; font-size: 10.5px !important; }
#pin-chat.ms-btn--on { border-color: rgb(var(--accent-rgb) / .45) !important; color: var(--accent) !important; background: rgb(var(--accent-rgb) / .08) !important; }
#delete-chat.ms-btn--armed { border-color: rgb(var(--danger-rgb) / .45) !important; color: var(--danger) !important; background: rgb(var(--danger-rgb) / .08) !important; animation: ms-throb 1.1s ease-in-out infinite; }

#settings-accordion {
    position: relative !important;
    flex: 0 0 auto !important;
    min-height: 46px !important;
    margin-top: 0 !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    overflow: visible !important;
    z-index: 4 !important;
}
#settings-accordion > .label-wrap {
    min-height: 46px !important;
    padding: 0 12px !important;
    cursor: pointer !important;
    border-radius: var(--radius-md) !important;
}
#settings-accordion > .label-wrap span { font-size: 12px !important; font-weight: 700 !important; }
#settings-accordion .wrap {
    padding: 12px !important;
    gap: 12px !important;
    max-height: min(58dvh, 560px) !important;
    overflow-y: auto !important;
}
#settings-accordion .gradio-container, #settings-accordion .form { gap: 10px !important; }
#settings-accordion .label-wrap + .wrap { border-top: 1px solid var(--line) !important; }
#settings-accordion label span { font-size: 11px !important; font-weight: 600 !important; }
#settings-accordion .info { font-size: 10px !important; line-height: 1.35 !important; color: var(--muted) !important; }
#seed-row { align-items: end !important; gap: 7px !important; }
#seed-field { flex: 1 1 auto !important; min-width: 0 !important; }
#seed-shuffle { flex: 0 0 76px !important; min-height: 38px !important; font-size: 11px !important; }
#reset-settings { min-height: 34px !important; font-size: 11px !important; }
.ms-note-box {
    margin: 0 !important; padding: 9px 10px !important;
    border: 1px solid var(--line) !important; border-radius: var(--radius-sm) !important;
    background: var(--surface-2) !important; color: var(--muted) !important;
    font-size: 10.5px !important; line-height: 1.45 !important;
}

/* ------------------------------ chat ----------------------------------- */
#chat-panel {
    position: relative !important;
    display: flex !important; flex-direction: column !important;
    flex: 1 1 0 !important; min-width: 0 !important; height: 100dvh !important;
    padding: 18px 22px 16px !important; gap: 10px !important;
    background: var(--bg) !important; overflow: hidden !important;
}
#chat-header {
    flex: 0 0 auto !important; align-items: center !important;
    min-height: 52px !important; margin: 0 !important; gap: 18px !important;
    padding: 0 2px !important;
}
#chat-heading { flex: 1 1 auto !important; min-width: 190px !important; }
#chat-heading h2 { margin: 0 0 3px !important; font: 700 20px/1.15 var(--display) !important; letter-spacing: -.025em; }
#chat-heading p { margin: 0 !important; max-width: 520px; color: var(--muted) !important; font-size: 11.5px !important; line-height: 1.4 !important; }
#header-controls {
    flex: 0 0 auto !important; display: flex !important; align-items: center !important;
    justify-content: flex-end !important; gap: 9px !important; margin: 0 !important;
}
#clinical-badge {
    display: inline-flex !important;
    align-items: center !important;
    min-height: 34px !important;
    padding: 0 11px !important;
    border: 1px solid rgb(var(--accent-rgb) / .22) !important;
    border-radius: 999px !important;
    background: rgb(var(--accent-rgb) / .07) !important;
    color: var(--accent) !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}
#image-type {
    width: auto !important; min-width: 300px !important; margin: 0 !important;
    padding: 3px !important; border: 1px solid var(--line) !important;
    border-radius: 12px !important; background: var(--surface) !important;
}
#image-type .wrap { display: flex !important; gap: 3px !important; padding: 0 !important; }
#image-type label {
    flex: 1 1 0 !important; min-width: 0 !important; margin: 0 !important;
    padding: 7px 9px !important; border: 0 !important; border-radius: 8px !important;
    justify-content: center !important; text-align: center !important;
    color: var(--muted) !important; background: transparent !important;
    font-size: 10.5px !important; font-weight: 600 !important; white-space: nowrap !important;
    transition: background .15s var(--ease), color .15s var(--ease) !important;
}
#image-type label:hover { background: var(--surface-2) !important; }
#image-type label.selected { background: var(--accent) !important; color: var(--on-accent) !important; box-shadow: 0 3px 8px -5px rgb(var(--accent-rgb) / .7); }
#image-type input[type="radio"] { position: absolute; opacity: 0; pointer-events: none; }

#meta-bar {
    flex: 0 0 auto !important;
    min-height: 24px !important;
    margin: 0 !important;
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    pointer-events: none !important;
}
.ms-tokenbar {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: 7px;
    min-height: 24px;
    padding: 0 2px;
    color: var(--muted);
    font-size: 9.5px;
    line-height: 1;
}
.ms-tokenbar__label { font-weight: 600; letter-spacing: .02em; }
.ms-tokenbar strong { color: var(--text); font-size: 9.5px; font-weight: 700; }
.ms-tokenbar__unit { opacity: .72; }
.ms-tokenbar__track { width: 44px; height: 3px; overflow: hidden; border-radius: 999px; background: var(--line); }
.ms-tokenbar__fill { display: block; height: 100%; border-radius: inherit; background: var(--accent); transition: width .2s ease; }
.ms-tokenbar--warn .ms-tokenbar__fill { background: var(--warn); }
.ms-tokenbar--warn strong { color: var(--warn); }

#scroll-anchor { position: relative !important; flex: 1 1 0 !important; min-height: 0 !important; overflow: hidden !important; margin: 0 !important; }
#conversation-view {
    height: 100% !important; min-height: 0 !important;
    border: 1px solid var(--line) !important; border-radius: var(--radius-lg) !important;
    background: var(--surface) !important; box-shadow: var(--shadow) !important;
    overflow: hidden !important;
}
#conversation-view .wrap { height: 100% !important; overflow-y: auto !important; overflow-x: hidden !important; padding: 22px clamp(14px, 4vw, 54px) 30px !important; scroll-behavior: smooth; }
#conversation-view .message-row { position: relative !important; margin: 0 auto 12px !important; max-width: 920px !important; }
#conversation-view .message { border: 1px solid var(--line) !important; border-radius: 15px !important; padding: 10px 13px !important; box-shadow: none !important; line-height: 1.55 !important; }
#conversation-view .user { background: var(--accent) !important; border-color: transparent !important; border-bottom-right-radius: 5px !important; box-shadow: 0 6px 16px -12px rgb(var(--accent-rgb) / .9) !important; }
#conversation-view .user, #conversation-view .user * { color: var(--on-accent) !important; }
#conversation-view .bot { background: var(--surface-2) !important; border-bottom-left-radius: 5px !important; }
#conversation-view .bot:has(.ms-cap), #conversation-view .bot:has(img) { background: transparent !important; border-color: transparent !important; padding: 2px !important; }
#conversation-view .message-row img { display: block; max-width: min(100%, 620px); max-height: min(55dvh, 520px) !important; margin: 4px 0 !important; border: 1px solid var(--line); border-radius: 13px; object-fit: contain; cursor: zoom-in; animation: ms-develop .6s var(--ease) both; }
#conversation-view .placeholder-content { max-width: 760px; margin: 0 auto; padding: 8px !important; color: var(--muted) !important; }
#conversation-view .placeholder-content h3 { color: var(--text) !important; font: 700 22px/1.2 var(--display) !important; }
#conversation-view .placeholder-content code {
    display: inline-block; margin: 4px 4px 0 0; padding: 7px 9px;
    border: 1px solid var(--line); border-radius: 8px; background: var(--surface-2);
    color: var(--text); cursor: pointer; transition: border-color .15s ease, background .15s ease, transform .15s ease;
}
#conversation-view .placeholder-content code:hover { border-color: rgb(var(--accent-rgb) / .45); background: rgb(var(--accent-rgb) / .08); transform: translateY(-1px); }

.ms-cap { margin-top: 7px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 9px; background: var(--surface-2); color: var(--text); font-size: 10.5px; line-height: 1.4; }
.ms-cap__meta { color: var(--muted); font-size: 9.5px; }
.ms-error { display: flex; align-items: flex-start; gap: 8px; padding: 9px 10px; border: 1px solid rgb(var(--danger-rgb) / .28); border-radius: 9px; background: rgb(var(--danger-rgb) / .07); color: var(--danger); font-size: 11px; }
.ms-error__icon { display: grid; place-items: center; width: 18px; height: 18px; border-radius: 50%; background: rgb(var(--danger-rgb) / .12); font-weight: 800; }
.ms-pending { display: flex; align-items: center; gap: 11px; padding: 10px 12px; border: 1px solid rgb(var(--accent-rgb) / .25); border-radius: 13px; background: rgb(var(--accent-rgb) / .06); }
.ms-pending__frame { position: relative; width: 30px; height: 30px; flex: 0 0 30px; border: 1px solid rgb(var(--accent-rgb) / .35); border-radius: 50%; overflow: hidden; }
.ms-pending__scan { position: absolute; left: -20%; top: 45%; width: 140%; height: 2px; background: var(--accent); box-shadow: 0 0 12px var(--accent); animation: ms-scan 1.1s ease-in-out infinite; }
.ms-pending__label { display: block; font-weight: 700; font-size: 11.5px; color: var(--text); }
.ms-pending__hint { display: block; margin-top: 2px; color: var(--muted); font-size: 10px; }

.ms-copy { position: absolute; right: 7px; bottom: -6px; opacity: 0; padding: 4px 7px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); color: var(--muted); font-size: 9px; cursor: pointer; box-shadow: var(--shadow); transition: opacity .15s ease, transform .15s ease; z-index: 3; }
.message-row:hover .ms-copy, .ms-copy:focus { opacity: 1; transform: translateY(-1px); }
.ms-copy.is-done { color: var(--accent); }

#scroll-controls { position: absolute; right: 20px; bottom: 16px; z-index: 5; pointer-events: none; }
.ms-jump { display: grid; place-items: center; width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 50%; background: var(--surface); box-shadow: var(--shadow); pointer-events: auto; cursor: pointer; opacity: 0; transform: translateY(5px); transition: opacity .18s ease, transform .18s ease; }
.ms-jump.is-visible { opacity: 1; transform: none; }
.ms-jump__arrow { width: 8px; height: 8px; border-left: 1.5px solid var(--text); border-top: 1.5px solid var(--text); transform: rotate(45deg) translate(2px, 2px); }

/* ---------------------------- composer --------------------------------- */
#composer { flex: 0 0 auto !important; gap: 5px !important; margin: 0 !important; }
#prompt-box {
    position: relative !important; display: flex !important; align-items: flex-end !important;
    border: 1px solid var(--line-strong) !important; border-radius: var(--radius-xl) !important;
    background: var(--surface) !important; box-shadow: var(--shadow-lg) !important;
    overflow: hidden !important; transition: border-color .2s ease, box-shadow .2s ease !important;
}
#prompt-box:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgb(var(--accent-rgb) / .12), var(--shadow-lg) !important; }
#prompt-box textarea { min-height: 48px !important; max-height: 150px !important; padding: 13px 15px !important; border: 0 !important; background: transparent !important; font-size: 13.5px !important; line-height: 1.45 !important; color: var(--text) !important; resize: none !important; }
#prompt-box .submit-button { flex: 0 0 auto !important; min-width: 92px !important; min-height: 36px !important; margin: 0 8px 8px 0 !important; padding: 7px 15px !important; border-radius: 11px !important; font-size: 11.5px !important; font-weight: 700 !important; box-shadow: 0 7px 14px -10px rgb(var(--accent-rgb) / .9) !important; }
#prompt-box .submit-button:hover { transform: translateY(-1px); }
#composer::after { content: "Enter to generate  •  Shift + Enter for a new line  •  Ctrl/Cmd + / focuses the prompt"; display: block; padding: 0 6px; color: var(--muted); font-size: 9.5px; line-height: 1.2; }

/* -------------------------- mobile backdrop ---------------------------- */
#mobile-menu { display: none !important; }
#mobile-backdrop { display: none; }

/* ---------------------------- lightbox --------------------------------- */
.ms-lightbox { position: fixed; inset: 0; z-index: 1000; display: grid; place-items: center; padding: 24px; background: rgb(0 0 0 / .72); opacity: 0; transition: opacity .2s ease; }
.ms-lightbox.is-open { opacity: 1; }
.ms-lightbox__panel { position: relative; max-width: min(94vw, 1100px); max-height: 92vh; padding: 14px; border: 1px solid rgb(255 255 255 / .12); border-radius: 16px; background: #0b1113; box-shadow: 0 30px 80px -30px rgb(0 0 0 / .9); }
.ms-lightbox__panel img { display: block; max-width: 100%; max-height: 78vh; object-fit: contain; border-radius: 10px; }
.ms-lightbox__close { position: absolute; top: 8px; right: 8px; z-index: 2; width: 34px; height: 34px; border: 1px solid rgb(255 255 255 / .15); border-radius: 50%; background: rgb(0 0 0 / .5); color: #fff; font-size: 22px; line-height: 1; cursor: pointer; }
.ms-lightbox__caption { max-width: 80ch; padding: 9px 3px 0; color: rgb(255 255 255 / .72); font-size: 10px; line-height: 1.4; }

/* ---------------------------- animation -------------------------------- */
@keyframes ms-develop { from { opacity: 0; filter: blur(10px); transform: scale(.985); } to { opacity: 1; filter: none; transform: none; } }
@keyframes ms-scan { 0%,100% { transform: translateY(-7px); opacity: .35; } 50% { transform: translateY(8px); opacity: 1; } }
@keyframes ms-pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: .45; transform: scale(.82); } }
@keyframes ms-throb { 0%,100% { box-shadow: 0 0 0 0 rgb(var(--danger-rgb) / .15); } 60% { box-shadow: 0 0 0 5px rgb(var(--danger-rgb) / 0); } }

/* ---------------------------- responsive ------------------------------- */
@media (max-width: 1100px) {
    #chat-panel { padding: 14px 15px 13px !important; }
    #chat-header { gap: 10px !important; }
    #image-type { min-width: 245px !important; }
    #clinical-toggle { padding: 0 8px !important; }
}

@media (max-width: 900px) {
    #app-shell { position: relative !important; }
    #app-sidebar {
        position: fixed !important; left: 0; top: 0; bottom: 0;
        width: min(86vw, 340px) !important; max-width: min(86vw, 340px) !important;
        transform: translateX(-102%); transition: transform .22s var(--ease);
        box-shadow: 18px 0 50px -25px rgb(0 0 0 / .55); z-index: 50;
    }
    #app-sidebar.is-open { transform: translateX(0); }
    #mobile-menu { display: block !important; }
    #mobile-menu button { min-width: 38px !important; min-height: 38px !important; padding: 0 !important; border-radius: 10px !important; }
    #mobile-backdrop.is-visible { display: block; position: fixed; inset: 0; z-index: 40; background: rgb(0 0 0 / .35); backdrop-filter: blur(2px); }
    #chat-panel { width: 100% !important; height: 100dvh !important; padding: 11px 10px 10px !important; }
    #chat-header { align-items: flex-start !important; }
    #chat-heading { display: grid !important; grid-template-columns: auto 1fr; column-gap: 9px; }
    #chat-heading h2 { font-size: 17px !important; }
    #chat-heading p { grid-column: 2; }
    #header-controls { width: 100% !important; flex-wrap: wrap !important; justify-content: flex-start !important; }
    #image-type { flex: 1 1 280px !important; min-width: 0 !important; }
    #clinical-toggle { flex: 0 1 auto !important; }
    .ms-meta { align-items: flex-start; padding: 7px 8px; }
    .ms-meta__left { flex-wrap: wrap; }
    .ms-meta__right { margin-left: auto; }
    #conversation-view .wrap { padding: 15px 10px 24px !important; }
    #conversation-view .message-row { max-width: 100% !important; }
    #composer::after { font-size: 9px; }
}

@media (max-width: 620px) {
    #chat-heading p { display: none; }
    #image-type { width: 100% !important; flex-basis: 100% !important; }
    #clinical-toggle { width: 100% !important; }
    #clinical-toggle label { font-size: 10.5px !important; }
    .ms-chip:nth-child(n+4) { display: none; }
    .ms-details { display: none; }
    .ms-meter__track { width: 34px; }
    #prompt-box textarea { min-height: 45px !important; font-size: 13px !important; }
    #prompt-box .submit-button { min-width: 76px !important; }
    #composer::after { content: "Enter to generate  •  Shift + Enter for a new line"; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .001ms !important; transition-duration: .001ms !important; }
}
"""

with gr.Blocks(
    title="Medsynth | Dermoscopy Image Studio",
    theme=gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        font=[
            gr.themes.Font("Instrument Sans"),
            gr.themes.Font("ui-sans-serif"),
            gr.themes.Font("system-ui"),
            gr.themes.Font("sans-serif"),
        ],
    ),
    css=APP_CSS,
    head=HEAD_HTML,
) as app:
    with gr.Row(elem_id="app-shell"):
        with gr.Column(scale=0, elem_id="app-sidebar"):
            gr.Markdown("# Medsynth\nDermoscopy image studio", elem_id="brand")
            model_status = gr.HTML(model_status_html(), elem_id="model-status")

            new_chat = gr.Button("＋  New conversation", variant="primary", elem_id="new-chat")

            with gr.Column(elem_id="history-form"):
                gr.HTML('<p class="ms-label">Conversations</p>', elem_id="history-label")
                history_search = gr.Textbox(
                    placeholder="Search conversations…",
                    show_label=False,
                    elem_id="history-search",
                    container=False,
                )
                history_empty = gr.HTML("", elem_id="history-empty")
                history_selector = gr.Radio(
                    choices=initial_choices,
                    value=initial_conversation_id,
                    show_label=False,
                    elem_id="history-list",
                )

            with gr.Column(elem_id="active-thread-panel"):
                gr.HTML('<p class="ms-label">Current conversation</p>', elem_id="manage-label")
                rename_title = gr.Textbox(
                    placeholder="Name this conversation",
                    value=initial_title,
                    show_label=False,
                    container=False,
                    elem_id="rename-title",
                    max_lines=1,
                )
                with gr.Row(elem_id="manage-row-1"):
                    pin_chat = gr.Button("Unpin" if initial_pinned else "Pin", variant="secondary", elem_id="pin-chat", elem_classes=["ms-btn--on"] if initial_pinned else [])
                    duplicate_chat = gr.Button("Duplicate", variant="secondary", elem_id="duplicate-chat")
                    export_chat = gr.Button("Export", variant="secondary", elem_id="export-chat")
                with gr.Row(elem_id="manage-row-2"):
                    clear_chat = gr.Button("Clear chat", variant="secondary", elem_id="clear-chat")
                    delete_chat = gr.Button("Delete", variant="secondary", elem_id="delete-chat")
                export_note = gr.HTML("", elem_id="export-note")

            with gr.Accordion("Generation settings", open=False, elem_id="settings-accordion"):
                steps = gr.Slider(
                    1, 50, value=int(initial_updates[2]), step=1,
                    label="Inference steps",
                    info="More steps can improve detail but increase generation time.",
                )
                guidance_scale = gr.Slider(
                    1, 15, value=float(initial_updates[3]), step=0.5,
                    label="Guidance scale (CFG)",
                    info="Higher values follow the prompt more strictly.",
                )
                with gr.Row(elem_id="seed-row"):
                    with gr.Column(elem_id="seed-field"):
                        seed = gr.Number(
                            value=int(initial_updates[4]), precision=0,
                            label="Seed", info="Use -1 for a new random seed.",
                        )
                    seed_shuffle = gr.Button("Shuffle", variant="secondary", elem_id="seed-shuffle")
                negative_prompt = gr.Textbox(
                    label="Negative prompt", value=initial_updates[5], lines=3,
                    info="Visual characteristics the model should avoid.",
                )
                gr.HTML(f'<p class="ms-note-box">{html.escape(ENHANCEMENT_HELP)}</p>', elem_id="enhancement-help")
                reset_button = gr.Button("Reset settings", variant="secondary", elem_id="reset-settings")

        with gr.Column(scale=1, elem_id="chat-panel"):
            gr.HTML('<div id="mobile-backdrop" aria-hidden="true"></div>')
            with gr.Row(elem_id="chat-header"):
                with gr.Row(elem_id="mobile-menu"):
                    gr.Button("☰", variant="secondary", elem_id="mobile-menu-button")
                gr.Markdown(
                    "## Skin image studio\n"
                    "Describe the image you want to synthesize. Results are synthetic and not for diagnosis.",
                    elem_id="chat-heading",
                )
                with gr.Row(elem_id="header-controls"):
                    gr.HTML(
                        '<span class="ms-header-badge">Clinical enhancement · always on</span>',
                        elem_id="clinical-badge",
                    )
                    image_type = gr.Radio(
                        choices=list(IMAGE_TYPES), value=initial_updates[0],
                        show_label=False, label="Image type", elem_id="image-type",
                    )
                # Always-on state: no user-facing switch can disable the trained prompt grammar.
                medical_prompt_enabled = gr.State(True)

            composer_context = gr.HTML(sync_context("", *initial_updates[:5]), elem_id="meta-bar")

            with gr.Column(elem_id="scroll-anchor"):
                chatbot = gr.Chatbot(
                    value=_render_conversation(initial_conversation_id),
                    type="messages",
                    height=None,
                    placeholder=(
                        "### Start a new synthesis\n\n"
                        "Describe the diagnosis, category, body site, skin type, and image view.\n\n"
                        "**Try a prompt:**\n\n"
                        "- `dermoscopy image of melanoma, malignant, back`\n"
                        "- `smartphone photo of basal cell carcinoma, malignant, face, Fitzpatrick skin type 2`\n"
                        "- `clinical photo of psoriasis, non-neoplastic, Fitzpatrick skin type 5`"
                    ),
                    show_label=False,
                    elem_id="conversation-view",
                    layout="bubble",
                    bubble_full_width=False,
                    sanitize_html=True,
                )
                gr.HTML(
                    '<button class="ms-jump" type="button" aria-label="Jump to latest" hidden>'
                    '<span class="ms-jump__arrow" aria-hidden="true"></span></button>',
                    elem_id="scroll-controls",
                )

            with gr.Column(elem_id="composer"):
                prompt = gr.Textbox(
                    placeholder="Describe the diagnosis, category, body site, skin type, and view…",
                    label="Message", show_label=False, lines=2, max_lines=6,
                    submit_btn="Generate", elem_id="prompt-box",
                )

    active_conversation = gr.State(initial_conversation_id)
    armed_delete = gr.State(None)

    context_inputs = [prompt, image_type, medical_prompt_enabled, steps, guidance_scale, seed]
    settings_inputs = [active_conversation, image_type, medical_prompt_enabled, steps, guidance_scale, seed, negative_prompt, prompt]
    conversation_outputs = [
        chatbot, active_conversation, history_selector, history_empty,
        image_type, medical_prompt_enabled, steps, guidance_scale, seed,
        negative_prompt, composer_context, delete_chat, armed_delete,
        history_search, pin_chat, rename_title, export_note,
    ]

    prompt.submit(
        generate_image,
        inputs=[prompt, active_conversation, image_type, steps, guidance_scale, seed, medical_prompt_enabled, negative_prompt],
        outputs=[chatbot, prompt, active_conversation, history_selector, history_empty, delete_chat, composer_context],
        show_api=False, show_progress="hidden",
    )

    new_chat.click(
        start_conversation, inputs=[history_search], outputs=conversation_outputs, show_api=False,
    ).then(lambda: gr.update(value=""), outputs=[prompt], show_api=False)

    history_selector.input(
        select_conversation, inputs=[history_selector, history_search], outputs=conversation_outputs, show_api=False,
    )
    delete_chat.click(
        arm_delete, inputs=[active_conversation, armed_delete], outputs=conversation_outputs, show_api=False,
    )
    clear_chat.click(
        clear_conversation, inputs=[active_conversation, history_search], outputs=conversation_outputs, show_api=False,
    )
    pin_chat.click(
        toggle_pin, inputs=[active_conversation, history_search], outputs=conversation_outputs, show_api=False,
    )
    duplicate_chat.click(
        duplicate_conversation, inputs=[active_conversation, history_search], outputs=conversation_outputs, show_api=False,
    )

    for event in (rename_title.submit, rename_title.blur):
        event(
            rename_conversation_from_ui,
            inputs=[active_conversation, rename_title],
            outputs=[history_selector, history_empty, rename_title],
            show_api=False,
        )

    export_chat.click(
        export_conversation, inputs=[active_conversation], outputs=[export_note], show_api=False,
    )
    history_search.change(
        filter_history, inputs=[history_search, active_conversation], outputs=[history_selector, history_empty, active_conversation], show_api=False,
    )
    seed_shuffle.click(
        randomize_seed, outputs=seed, show_api=False,
    ).then(
        persist_settings, inputs=settings_inputs, outputs=[composer_context], show_api=False,
    )
    reset_button.click(
        reset_settings,
        inputs=[active_conversation, prompt],
        outputs=[image_type, medical_prompt_enabled, steps, guidance_scale, seed, negative_prompt, composer_context],
        show_api=False,
    )

    commit_event = {gr.Slider: "release", gr.Textbox: "blur"}
    for control in (image_type, medical_prompt_enabled, steps, guidance_scale, seed, negative_prompt):
        event = getattr(control, commit_event.get(type(control), "change"))
        event(persist_settings, inputs=settings_inputs, outputs=[composer_context], show_api=False)

    prompt.input(sync_context, inputs=context_inputs, outputs=[composer_context], show_api=False)

    status_timer = gr.Timer(2.0)
    status_timer.tick(model_status_tick, outputs=[model_status], show_api=False)


def _warm_up():
    """Load the model in the background so the first prompt is not slow."""
    try:
        with _generation_lock:
            load_pipeline()
        _model_state["state"] = "ready"
        _model_state["detail"] = _describe_device()
    except Exception as error:
        print(f"Model warm-up failed: {error}")
        _model_state["state"] = "error"
        _model_state["detail"] = str(error)[:120]


if __name__ == "__main__":
    if model_is_available():
        print(f"Using trained model: {MODEL_DIR}")
        Thread(target=_warm_up, daemon=True).start()
    else:
        print(
            f"\nNo trained model found at {MODEL_DIR}.\n"
            "Run `python train.py --stage all` first (see README.md).\n"
            "The app will still open; generating will explain what is missing.\n"
        )
    app.queue().launch()
