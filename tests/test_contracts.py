import numpy as np
from PIL import Image

from contracts.messages import (
    AudioChunk,
    AudioOut,
    LlmToken,
    ToolCall,
    ToolResult,
    ToolSpec,
    Transcript,
    TurnState,
    VadEvent,
    VideoFrame,
    frame_to_data_url,
    frame_to_image_part,
)


def test_audio_chunk_roundtrip():
    chunk = AudioChunk(pcm=b"\x00\x01", sample_rate=16000, t_capture=1.0)
    assert chunk.sample_rate == 16000


def test_vad_event_kinds():
    assert VadEvent(kind="speech_start", t=0.0).kind == "speech_start"
    assert VadEvent(kind="speech_end", t=0.0).kind == "speech_end"


def test_transcript():
    t = Transcript(text="oi", is_final=True, t_start=0.0, t_end=0.1)
    assert t.is_final


def test_video_frame_to_data_url_pil():
    img = Image.new("RGB", (8, 8), color="red")
    frame = VideoFrame(image=img, frame_id=0, t=0.0)
    url = frame_to_data_url(frame)
    assert url.startswith("data:image/jpeg;base64,")


def test_video_frame_to_data_url_ndarray():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = VideoFrame(image=arr, frame_id=1, t=0.0)
    url = frame_to_data_url(frame)
    assert url.startswith("data:image/jpeg;base64,")


def test_frame_to_image_part_shape():
    img = Image.new("RGB", (4, 4))
    part = frame_to_image_part(VideoFrame(image=img, frame_id=0, t=0.0))
    assert part["type"] == "image_url"
    assert "url" in part["image_url"]


def test_tool_spec_as_openai_tool():
    spec = ToolSpec(name="foo", description="d", parameters={"type": "object"}, timeout_ms=1000)
    tool = spec.as_openai_tool()
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "foo"


def test_tool_call_and_result():
    call = ToolCall(id="c1", name="foo", arguments={"x": 1})
    result = ToolResult(tool_call_id=call.id, name=call.name, content="ok", is_error=False, t=0.0)
    assert result.tool_call_id == "c1"


def test_llm_token_and_audio_out():
    tok = LlmToken(text="a", is_tool_call=False, done=False)
    assert not tok.done
    out = AudioOut(pcm=b"123", is_last=True)
    assert out.is_last


def test_turn_state_enum_values():
    assert {s.value for s in TurnState} == {"IDLE", "LISTENING", "THINKING", "SPEAKING"}
