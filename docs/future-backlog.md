# DIY-Key future backlog

Work identified while bringing the AI Key features up on macOS against Protect 7.3.74
(November 2026). `BACKLOG.md` remains the list of short-term items; this is the longer view.

**Priority:** P1 can produce wrong results today. P2 is a clear improvement. P3 is optional.

**Evidence** is what the idea rests on: *firmware* and *Protect* mean read from AI Key 2.2.8 or
Protect's code, *live* means observed on a running instance.

## Not yet proven live

These are built and unit tested, but have not been seen working end to end.

| Feature | What is missing |
|---|---|
| License plates | No plate result has been uploaded yet |
| AI enhanced alarms | Never seen firing from Alarm Manager |
| Tap-to-find (`:7445/segment-video`) | Protect requested it before it existed; not called since |
| `fastReverification`, `enhanceImageGroup` | Protect has never sent them |
| Face reads and re-ID on the upgraded instance | Confirm after the host move |
| Several AI Keys on one console | Search-host contention was seen with an unadopted key |

## Accuracy

| Pri | Item | Evidence | Approach |
|---|---|---|---|
| P1 | Second-stage verification can reject real detections | firmware: no "nothing there" class; keeps the detection unless top score >= 99.5%. diy-key marks it invalid at 5% when its detector sees nothing | Default to keeping the detection, change type only on very strong evidence, use the firmware's confidence formula (`BACKLOG.md`) |
| P1 | Boxes about 1 s off moving objects on ONVIF cameras | live: on Driveway the box sits ahead of a passing car on every frame checked | Find where the offset enters: `diy-port` timestamps, restream delay, or recording. Crops, tags and faces on moving objects all depend on it |
| P2 | Per-frame capture time from `ubv` exports | firmware seeks by offset; `ubv` has clock-sync records | Only if an offset remains after the fix above |
| P2 | Face quality gate | firmware has `VerifyFaceQuality`, `EstimateFaceQuality`, blur and pose checks | `DIYKEY_FACE_MIN_PX` (default 40) is a stopgap; tune it, then consider blur and pose |
| P2 | Re-ID scale | live: same camera 23.5, across cameras 27.2-33.4, Protect's limit is 20-30 | Measure distances between different people first, then choose a scale from both |
| P3 | Real person outlines for tap-to-find | firmware uses a segmentation model; diy-key returns rectangles | Needs a segmentation model |
| P3 | Clothing colours | RAM++ does not produce them | Colour from the crop, or another attribute model |

## Summaries and scene awareness

| Pri | Item | Evidence | Approach |
|---|---|---|---|
| P1 | Event-aware summaries | firmware sends one generic prompt and one frame; live: passing cars summarised as "Red car parked in driveway" | Crop to the triggering object and state its type (`DIYKEY_VLM_EVENT_CONTEXT`) |
| P2 | Stationary-object memory per camera | no precedent in firmware or Protect | Learn what stays put, drop it from tags and summaries, report when it moves. People are never hidden. Fixtures after 7 days (`DIYKEY_SCENE_MEMORY`) |
| P2 | Stronger summary model for wide views | live: "Pile of wood in yard" for a person in Backyard | Compare other MLX models on real clips, at acceptable latency |
| P3 | Feed `diy-port`'s tracker into the memory | `diy-port` already tracks parked and still objects continuously | Better than event-only learning, on its cameras only |
| P3 | Condensed history and daily digests | Protect has no digest feature or message channel | Scheduled condensing, for reading only, never fed back into detection |
| P3 | Webhook notifications | the AI Key interface can only attach text to an event | One setting for digests and "parked car moved" alerts |

## Throughput and reliability

| Pri | Item | Evidence | Approach |
|---|---|---|---|
| P2 | Stay available when the host sleeps | live: a sleeping Mac lost 100+ events in a day | Warn at startup on battery or closed lid, document `pmset`, surface it in logs |
| P2 | Separate the summary step from tagging | live: about 25 s tagging and 20 s summary per event, one worker | Low-priority summary queue, so a slow summary doesn't delay the next event |
| P3 | More than one worker | model locks limit overlap; untested live | Try 2 workers and measure on the shared GPU |
| P3 | Capacity versus a real key | FAQ says 1,000 per hour; diy-key manages about 70-120 | Only matters on a busy site |
| P3 | Queue lost on restart | in-memory; Protect marks those tasks failed | Probably acceptable |

## Packaging

| Pri | Item | Approach |
|---|---|---|
| P2 | RAM++ export script in the repo | There is no ready-made download. Add a tested `scripts/` export from the published checkpoint, pinned to recognize-anything `7cb804a` |
| P2 | One model setup step | A script that fetches OSNet, GFPGAN and the face models and writes the paths |
| P2 | Update the feature table in `docs/macos.md` | It still lists face, LPR, enhancement and summaries as missing |
| P3 | Weights licensing note | OSNet code is MIT, but its weights were trained on the MSMT17 research dataset; check terms before redistributing |

## Speech to text: build plan

Out of scope for now; this is the plan if it is picked up. Priority P3.

Dependency (2026-11-12): cameras paired to diy-port send no audio events, so Protect never
asks the AI Key to transcribe them. diy-port's backlog has `alrmSpeak` via EventSmartAudio
(Olli0103/unifi-protect-local-ai `aiport_audio.py` shows Protect 7.3.60 accepts it from an AI
Port and then queues the AI Key's `speechToText`). Olli's project also has a working speech
path to compare against.

How the firmware does it (`syswrapper.sh`, `ui-stt-agent` in AI Key 2.2.8): the camera detects
speech, Protect sends `speechToText`, the clip is fetched and its start read from the
`x-timestamp` / `x-start-timestamp` response headers, `ffmpeg` extracts 16 kHz mono WAV, and
whisper.cpp with a voice-activity model transcribes it. A signal-to-noise calculation and a
language-confidence threshold of 0.75 apply. The result goes to `/aiprocessors/speech-to-text`.

1. **Read the contract:** Protect's `speechToText` dispatch (`851272`) and the
   `/aiprocessors/speech-to-text` schema, plus the firmware's `post_enable_steps_speech_to_text`
   and `speech_to_text`, for the exact request and result shape. Not yet read in detail.
2. **Choose the engine:** whisper.cpp or MLX Whisper on Apple Silicon, with a voice-activity step.
   Compare accuracy and speed on real speech clips from a camera. English only, like the FAQ.
3. **Module** `stage3_wss/stt.py`, enabled by `DIYKEY_STT_MODEL`: WAV extraction, voice activity,
   transcription, and timestamps offset from the clip's true start (reuse `clip_fetch`).
4. **Wire it in:** take `speechToText` out of `ACK_ONLY_ACTIONS`, run it on the task queue,
   count it in `UI_AUDIO_RAM`, and advertise the feature flag only when the model is configured.
5. **Quality gates:** drop silent or low-signal clips and low-confidence languages rather than
   sending invented text, as the firmware does.
6. **Tests and live check:** unit tests for the contract, timestamps and gates. Then confirm that
   Protect sends the request once the feature is on, and that "show transcript" appears on a
   real speech event.

**Open check:** the FAQ is inconsistent. Its comparison table lists speech to text for all camera
types (G3/ONVIF through an AI Port), while its text says AI-series cameras only. What matters is
whether a camera produces speech smart detections: a Doorbell speech event (`smartAudioDetect`,
`alrmSpeak`) was seen live on 2026-11-08. Confirm early that Protect sends `speechToText` for it.

## Decided against, for now

- **Speech to text:** out of scope for now; see the build plan above.
- **Deep Understanding:** needs a hidden Protect setting; left off on purpose.
- **Running the firmware's models:** they are compiled for NVIDIA hardware and cannot run on a Mac.
