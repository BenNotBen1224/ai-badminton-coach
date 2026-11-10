# Preview speed

Paths under `obj2/work/` name files in the lab notebook. They are not part of this repo.

NOT MET. Warm preview on clip 03 is 3.59 s, not under 1 s. The same sampler also misses the quality gate on the phase-1 development reference. This is not a sealed acceptance.

## Kept clock

One quiet run, pose slots 4, two decode contexts, one thread each, pose follow. Wall 3.5887 s. Stall 0.18 s. Source hash `5987157214309241cb1456d14a8cfdf723e7dc772d4f34122dc61dbe4a6ebc27` is the older file that was timed. The tree now also has the window-identity fix. That fix was not part of the 3.59 s run. Do not treat later walls as this number.

| try | wall | kept |
| --- | ---: | --- |
| 4 pose slots | 3.59 s | yes, one quiet run |
| 3 decode contexts | 3.67 s | no |
| 2 decode threads | 3.70 s | no |
| 2 pose groups | 5.43 s | no |
| flow, four frames at a time | 4.14 s | no |

The flow run was quiet (stall 0.18 s), hash `80512178ddbcd2d02a3f8768db5a4c306f6b706bfe7f401712d7a77f44f3bb24`. Pose time fell, but the wall rose and crops rose from 413 to 486. `FOLLOW` is back to `pose`.

## Quality, clip 03 only

Scorer: `turbo.score.score_pair`, with repeat and phase-0 exclusions. Reference: `turbo/refs/validation-03.json`, development data, not phase 2. 360 of 1080 frames excluded. 0 consecutive repeats.

| output | PCK10 | fast PCK10 | found | PCK05 |
| --- | ---: | ---: | ---: | ---: |
| dense step4, same reference | 0.9904 | 0.9797 | 1.000 | 0.9734 |
| pose-follow preview | 0.9407 | 0.8873 | 1.000 | 0.8175 |
| flow preview, rejected | 0.8965 | 0.8046 | 0.997 | 0.7835 |

The pose-follow document is `obj2/work/S/preview-03-womens-singles-rally-gop-2.json`, same sampler as the kept clock, not a re-score of hash `59871572`. It already misses both PCK limits, before the reverted outside-feet skip. Found holds. Flow is worse.

Same-bucket deltas versus dense step4: all-frames PCK10 −0.0497 (gate −0.030) and fast PCK10 −0.0925 (gate −0.060). An earlier −0.1031 figure subtracted pose fast from dense all-frames PCK10, not from dense fast. Both figures fail. Per-role rates are in `obj2/work/S/score-clip03-roles.json`.

| role | PCK10 delta | fast PCK10 delta | found delta | PCK05 delta |
| --- | ---: | ---: | ---: | ---: |
| near | −0.0404 | −0.0425 | 0 | −0.1505 |
| far | −0.0590 | −0.1425 | 0 | −0.1613 |

Both roles miss the all-frames PCK limit. The fast miss is mostly the far player.

## Inference hook

The old census patched `turbo.frozen.ovcache` while the service calls `ovcache.infer`, and detection had already bound `infer` in `detect.py`. A corrected hook on those live modules, after warm-up, saw 413 pose calls and 37 detector calls. Those match the tracker's 413 crops and 37 detections. Shapes were `(1, 256, 192, 3)` and `(1, 384, 640, 3)`. No other calls.

Python callers are blocked in `request.wait`: 0.08 s of caller CPU against 19.3 s of summed wait. That sum overlaps and is not a critical path. 445 of 450 waits were blocked. Side samples, about 0.30 s apart, put 38.8 CPU-s on unnamed threads while a wait overlapped the sample and 5.9 CPU-s when none did. The recorded decode thread added none of that. This is not proved model math. The published single-thread pose cost for 413 crops is 4.9 CPU-s, and that number is not a measurement of this run. Extra runtime threads, the detector, unnamed decode threads, and pool spin all remain possible.

The check recorded total system CPU median 72%, max 88%. That number includes this process. It is not external contention. Wall 9.91 s is not a timing result. A later note that the paused viewer made no draw calls does not prove the machine is quiet. Acceptance still needs the existing CPU check. Source `service.py` SHA256 `98474cfab7e21fb8cd46829b2c0cda64362d20aa5e63f1e19f71ef75651bc207`. An earlier hook that scanned every thread on each call took 221.7 s and its CPU split is invalid: `obj2/work/S/validate-infer-expensive.json`.

Skipping pose when feet leave the court was scored and reverted. Crops fell from 413 to 333. PCK10 delta became -0.084, fast PCK10 -0.142, found -0.035. Evidence: `obj2/work/S/score-clip03-outside.json`. The live path does not skip those tracks.

## Same-frame pose batch

Not adopted. The stacking path is removed from `service.py`. One request per crop stays, still launched from Python threads. Script `obj2/work/S/probe_pose_batch.py` refuses to rerun. Result `obj2/work/S/probe-pose-batch.json`. The stored 83% is total system CPU, including this process, not external load. `load_comparable` is withdrawn. Not a timing result and not acceptance.

| path | requests | wall | note |
| --- | ---: | ---: | --- |
| one crop at a time, first | 413 | 9.489 s | also copied 40 frames of crops; that copy is not a 4.6 s gap |
| one crop at a time, second | 413 | 4.892 s | same load median as the first |
| one request for the frame | 156 | 4.592 s | 74 pairs, 63 triples, 19 quads; crops stayed 413 |

The two one-at-a-time walls differ by 4.6 s. The 0.30 s stacked dip is inside that spread. A script flag that said adopt passed by 0.9 ms and is rejected. Pose clocks overlap and are not a critical path; those also differ by 1.7 s between the two one-at-a-time runs.

Same saved crops, warmed, 20 repeats: 4.131 ms per crop one at a time, 4.309 ms stacked. Stacking did not make the work cheaper.

Rows matched. 2810 players. No identity, box, source, or role changes. Isolated joints matched. One rounded confidence differed by 0.0001. This run was not scored against the reference. The earlier pose-follow miss still stands. The run used `service.py` SHA256 `cbf6be4d3ac63855119c4fd0b9574d8bfe85a67b8bc4a85251a41cf183ce8b65`. After that run the stacking path was removed. That removal left hash `919e0e2fd1163de60a1240d7fba451615af93d51e7fbfa27145ae969ca264e1f`. A later error-propagation edit changed the file again. Do not present 4.59 s as the kept clock.

## CPU labels

`psutil.cpu_percent` in the old samplers is total system CPU, including this process. The stored medians stay in the old JSON files. They do not establish external contention, and they do not prove unnamed CPU was useful inference.

| report | stored median | what it is |
| --- | ---: | --- |
| census-contended.json | 35.7% | total system CPU |
| census-flow.json | 35.7% | total system CPU |
| validate-infer.json | 72.3% | total system CPU |
| validate-infer-expensive.json | 46.1% | total system CPU; 221.7 s run stays invalid |
| probe-pose-batch.json | 83.4 / 83.5 / 80.4% | total system CPU; `load_comparable` withdrawn |

Correction file: `obj2/work/S/cpu-label-correction.json`. The quiet pre-run check is unchanged.

## Native async launch

Not adopted. Production still starts two Python threads per pose wave. Script `obj2/work/S/probe_infer_async.py`. Result `obj2/work/S/probe-infer-async.json`. Source `service.py` SHA256 `919e0e2fd1163de60a1240d7fba451615af93d51e7fbfa27145ae969ca264e1f`.

Saved crops matched exactly: 413 starts, 0 px joint move. Full documents matched on three pairs. Failure handling waited the started request and raised the real error.

Same crops, three repeats: 4.68 ms per crop with threads, 3.80 ms with native async. Those repeats do not overlap.

Full preview, interleaved: thread 4.82 / 6.11 / 5.82 s, async 6.04 / 5.49 / 5.71 s. Median gap 0.12 s. The thread reruns already differ by 1.29 s. Process CPU was 37–44 s either way. Total system CPU median was 85–93%, including this process. Not external load, not a timing result, not acceptance. Quality was not re-scored. The pose-follow miss still stands.

## Not measured

Rows at a 0.75 s pose budget and a 0.95 s pose budget were not run. They are not filled in.

## Window identity

Separate from the clock. `python -m unittest tests.test_s_window_ids -v` from `turbo`, comparison venv. First run failed `1 != 0`. After the overlap mapping, OK in 0.619 s. Controlled detections, not a real-video identity check. Do not time that test again. Source at that check began `727899e8`.

## Sampling diagnosis

NOT MET. No tested global sampling point meets both the one-second budget and the quality gates. A setting between these two points was not run. That middle setting is untested. It is not proved impossible. Phase 0 only for this diagnosis. Phase 1 was not used to fit a threshold. Sealed data was not opened.

The every-4th diagnosis and the every-reference ceiling were measured on `service.py` SHA256 `919e0e2fd1163de60a1240d7fba451615af93d51e7fbfa27145ae969ca264e1f`. A later error-propagation fix changed that file. It was not part of either wall.

The current every-4th preview is not the final operating point. Its kept clock is still the older 3.5887 s run on hash `59871572`. The 9.0001 s ceiling is a different source and a different sampler. Those two walls are not a paired speed comparison. Do not replace 3.5887 s with 9.0001 s.

Same phase-0 frame IDs, clip 03, dense step4 against the retained every-4th preview. Judge is `fast/runs/judge-03.json`. Script `obj2/work/S/probe_sample_diag.py`. Result `obj2/work/S/sample-diag-phase0.json`. Numbers below are the stored observations.

| set | frames | PCK10 delta | fast PCK10 delta |
| --- | ---: | ---: | ---: |
| posed | 52 | -0.017 | -0.021 |
| filled in | 308 | -0.054 | -0.099 |

Dense PCK10 on those two sets was 0.989 and 0.991. The larger filled-in deficit supports an interpolation explanation. It does not isolate the whole cause, because the posed and filled-in subsets are different frames. The remainder is a difference of those deltas, not a separated cause. The far player on fast filled-in frames is -0.145. Found stays 1. The GOP keep list matches the 156 measured frames. A serial listing does not; that was a decoder difference, not missed detections.

One denser ceiling: pose every reference frame, this process only. One run. Not acceptance. Wall 9.0001 s. 1502 crops. 563 frames kept. Process CPU 70.2 s is the analyze call only. The 64.2% mean system CPU starts before service construction and warmup, so that mean includes those phases, not only the timed analyze call. It is total system CPU, including this process, not external load. Pre-check system CPU 8.1%. Result `obj2/work/S/sample-ceiling-phase0.json`.

| set | frames | PCK10 delta | fast PCK10 delta |
| --- | ---: | ---: | ---: |
| posed | 194 | -0.003 | -0.005 |
| filled in | 166 | -0.003 | -0.007 |

The ceiling deficits are small on both sets. That is consistent with less fill-in. It is not proof that interpolation was the whole earlier gap. The far fast filled-in delta is -0.012. Pose work summed to 10.8 s and overlaps decode, so that sum is not the critical path.

Chart: `obj2/work/S/sample-pareto.html`. Corrected reading: `obj2/work/S/sample-claim-limits.json`. Raw JSON numbers were not rewritten.

## Error propagation

Production `_infer_crops` joined a failed worker and returned a list with `None` beside a successful row. Reproduced on that helper before the fix: an infer failure returned `[row, None]`; a decode failure returned `[None, row]`; the next wave still started.

The helper now joins every started request in the wave, then raises the real error. It does not return those rows. A later wave does not start. Normal request count, order, and sampling are unchanged. One request per crop stays.

Test: `python -m unittest tests.test_s_infer_error -v` from `turbo`, comparison venv. 5 tests, OK. A wave is checked as a request multiset, not as call order. Returned row order and count stay literal. The rejected async probe does not cover this helper. Captured command, exit code, and hashes: `obj2/work/S/infer-error-pass.json`.

Source `service.py` SHA256 `e319791f323f1803a7d8dabb0e8d45eff2465de25aba7a0cb061bed5b6bd2d20`. Not a timing result. The kept clock is still 3.5887 s. Quality is still NOT MET.

## INT8 pose check

Not adopted. No production change. The checkpoint, `service.py`, `fast\`, and the existing venv were not modified. Calibration used 135 clip-03 phase-0 crops from the retained preview boxes, not reference labels. Sealed data and clips 08/09/10 were not opened.

NNCF quantized the built graph. 136 FakeQuantize nodes. Same uint8 input. Five interleaved repeats, two requests per wave, four streams. Native calls 1350, matching the crop count. Result `obj2/work/S/probe-pose-int8.json`.

FP32 was 2.71 to 30.75 ms/crop. INT8 was 2.44 to 6.64. The ranges overlap. The median implied saving on 413 crops is -0.31 s. That is not beyond repeat variation, and it is inside the known 1.29 s full-preview spread. The 30.7 ms FP32 repeat sat at 99.6% total system CPU, which includes this process and is not external load. Dropping that spike still leaves overlapping ranges. The quiet gap is about 0.12 s on 413 crops. Pose time overlaps decode, so that figure is an upper bound, not a critical path.

Output shapes match, `(1, 17, 384)` and `(1, 17, 512)`. A swap does not line the first joint up. Mean joint error is 104 px. 99.8% of joints move more than 2 px. That is not an output-order bug.

Untimed control, `obj2/work/S/probe-pose-int8-control.json`. `production_build` with no NNCF matches `pose.Pose` on 2 near and 2 far saved crops. Both sides name `simcc_x` `(1, 17, 384)` f32 and `simcc_y` `(1, 17, 512)` f32. Pairing is that axis length, then the same `pose.decode`. Tensor max abs is 0. All 17 joints match. Max xy is 0 px. The 104 px gap is not a preprocessor or decoder mismatch. It remains a difference between the INT8 model and a graph that matches `pose.Pose`.

The control JSON flag `integer_kernels_proved` is too coarse. Any i8 or u8 output set it. FakeQuantize only proves conversion. The compiled runtime graph, `obj2/work/S/int8-runtime-precision.json`, exposes integer JIT primitives for all 38 convolutions, all 5 fully connected layers, both matmuls, and all 3 pooling nodes. 34 of 47 fused subgraphs still use a float JIT. Output element type is not the kernel precision: 37 of 38 convolutions output f32 while their runtime precision is u8. The inference precision hint is float32. This does not adopt INT8.

No new quantization. No full preview. A middle sampling point remains untested. Service hash is unchanged. The one-second and quality gates stay NOT MET.

## Batch-8 pose check

Not adopted. INT8 stays rejected. No production change.

The ONNX batch dimension is already dynamic. `model.reshape` to 8 needs no hand edit. OpenVINO folds some `ShapeOf` ops into constants and drops one `Transpose`. The first run stopped on that op-type list; that stop is kept in `obj2/work/S/probe-pose-batch8-reshape-stop.json`. It is not the comparison.

The comparison is the same 135 saved phase-0 crops. Both sides use throughput, 4 streams, threads left unset (compiled as 16), and 2 requests in flight. The wave side is one crop per request, 135 calls, no padding. The batch side is 8 crops per request, 17 calls, 1 padded slot. In-flight crops are 2 versus 16. That crop difference is the batch, not extra streams. Timer is `ovcache.infer` only. Decode is outside it. Five alternating repeats. Result `obj2/work/S/probe-pose-batch8.json`.

Wave walls were 0.457 to 0.480 s, median 0.461 s. Batch-8 walls were 0.339 to 0.365 s, median 0.351 s. The ranges do not overlap. Process CPU was about 2.9 to 3.5 s on the wave side and 2.2 to 2.6 s on the batch side, so the saving is not only overlapped wait. Total system CPU in the log is 62 to 79 percent and includes this process. It is not outside load.

Semantic `simcc_x` / `simcc_y` and the inverse-transformed joints match. Tensor max abs is 0. Max xy is 0 px on all 135 crops, including every timed repeat.

The implied saving on 413 crops is 0.336 s. The pre-set bar was more than the 1.29 s preview spread. This misses that bar. Pose also overlaps decode, so 0.336 s is an upper bound, not a preview gain. The remaining 0.35 s is still the model work on several cores. Cutting 135 calls to 17 removed the smaller part. Not promising. No pipeline change.

## Pose thread cap

Not adopted. Batch-8 and INT8 stay unused. No production change.

The old latency run is not this pair. It also set LATENCY and 2 pose slots. This run changed only the compiled pose thread count: unset, which compiled as 16, versus 4. Both sides stayed throughput, 4 streams, 4 pose slots, detector at 2 latency threads, pose follow, and GOP decode. Separate scratch caches. `turbo/cache` and `service.py` were not written. Service hash `e319791f323f1803a7d8dabb0e8d45eff2465de25aba7a0cb061bed5b6bd2d20`.

Clip 03, five alternating warmed preview pairs, plus one warmup pair. The pre-run total system CPU stayed above 10% for 89 seconds, so this is a contended paired diagnostic, not a quiet acceptance. System CPU during the calls includes this process.

Walls overlapped. Current 3.429 to 3.538 s. Four threads 3.469 to 3.983 s. Pair deltas changed sign. Median delta -0.133 s, four-thread side slower on that median, and not outside this run's variation. No measurable wall difference.

Process CPU did separate: about 25.1 to 26.3 s versus 19.2 to 21.5 s. Summed pose time, which overlaps and is not the wait, rose from about 2.45 to 2.76 s to about 3.87 to 4.41 s. Decode stayed near 2.1 s. The extra threads were off the user-wait path on this clip. That does not prove they were only spinning.

All 12 documents are the same bytes. Joints, confidence, boxes, roles, and missingness match. 413 crops and 156 pose calls on every run. The 156 is crop groups: `pose_calls` increments once per `_infer_crops` group. Native requests stay one per crop, 413. The old JSON fields are unchanged. Result `obj2/work/S/probe-pose-threads.json`. The one-second goal stays NOT MET.

## Crop wait

Not a candidate. No production change. The old 0.21 ms figure is a repeated identical crop. It is not the live crop.

Live retrieve is almost all `to_ndarray`. Alloc is 0.04 ms, the blit is 0.09 ms, and `warpAffine` is 0.21 ms after the timer stops. On the one instrumented clip 03 run, 413 pose reformats had a 7.424 ms median. 62 of those had no other instrumented reformat, infer, or warp in flight and were still 5.277 ms. GOP 2 pose reformat summed to 0.817 s. That run's last actor was GOP 4. It does not replace the retained 10 of 10 GOP-2-last census.

A clean process, with no service and no OpenVINO, replayed the recorded GOP 2 rectangles. Same crop, 200 times: 0.180 ms. Forty different frames, one fixed box: 0.199 ms. The same frame and the 104 recorded boxes, a new size every call: 3.681 ms. GOP 2 pose crops change size on 103 of 103 consecutive calls, 72 unique sizes. Walking frames is not the cost. Changing the crop size is.

The old line that the 8 ms call is waiting, not conversion, used the reused-size baseline. About 3.7 ms is the conversion of a new size. The rest of the live 7.4 ms is not isolated. The no-overlap calls are still about 1.6 ms above the clean size-change median. Windows thread time moves in 15.625 ms steps, so per-call CPU cannot prove that remainder is a wait.

OpenCV reported 16 threads before, during, and after the preview. No turbo code sets that count. It does not explain retrieve: the OpenCV warp is outside the timer and is 0.21 ms. A scratch `threads=1` reformat of the reused crop was 0.039 ms versus 0.127 ms for the default. Both are under a millisecond. That is not a candidate.

PyAV keeps one reformatter per thread. Production does not pass a thread count, so the default is 0, automatic. A size change is enough to cause the 3.7 ms. This run did not trace a graph rebuild inside the bundled swscale.

Evidence: `obj2/work/S/probe-retrieve-split.json`, `obj2/work/S/probe-retrieve-alone.json`. Service hash remains `e319791f323f1803a7d8dabb0e8d45eff2465de25aba7a0cb061bed5b6bd2d20`. One second and quality stay NOT MET.

## Conversion threads

Not adopted. No production change. This is the changing-size set, not the reused-size 0.039 ms versus 0.127 ms scratch.

The only difference was `to_ndarray(..., threads=1)` against production, which passes no thread count (PyAV treats that as 0, automatic). Same blits, color metadata, and ordered 113 GOP 2 rectangles, including the 104 pose crops that change size on 103 of 103 consecutive calls. Five alternating warmed repeats. The timer is the whole helper. The array copy is outside it. Two persistent threads, two reformatter ids. Result `obj2/work/S/probe-sws-threads.json`.

Helper sum, median of five: 0.438 s current (0.434 to 0.443) versus 0.027 s threads=1 (0.0264 to 0.0268). Median paired gap 0.412 s. Ranges do not overlap. Every returned BGR array matched. The Python field on the reformatter was unreadable (`AttributeError`), so this is the `threads=1` call, not a census of the swscale context. It still does not prove a graph rebuild inside swscale.

Because that helper gap was outside its own variation, five alternating warmed clip 03 phase-0 previews ran on scratch caches. Current wall median 3.453 s (3.395 to 3.556). threads=1 median 2.814 s (2.694 to 2.826). Median paired gap 0.701 s. Ranges do not overlap. All 12 documents are the same bytes. Joints, confidence, boxes, and missingness match. 413 crops, 156 crop groups, 37 detections. The 156 is still groups, not native requests.

The retrieve figure in that JSON is a sum across actors, not the wait. It fell from about 3.35–3.47 s to 0.23–0.26 s. The wall fell only 0.70 s, so most of that sum was overlapping other work. Pose sums still overlap (about 2.45–2.74 s versus 2.04–2.46 s). Decode stayed near 2.1 s. Process CPU separated: about 24.9–27.7 s versus 20.1–21.5 s. That fits less conversion work, not a skipped analyze.

Quiet pre-check was 7.9%, 9.2%, 5.9% total system CPU, including this process. During the previews that total was 53–76%, including this process. It is not external load. OpenCV reported 16 threads. It was not changed and is not this lever.

The 2.814 s wall is not under 1 second. These bytes match the current preview, so the stored phase-1 miss still applies: all-frames PCK10 −0.0497 and fast PCK10 −0.0925. This slice did not re-score. Service, decode, turbo cache, weights, and `prepared-models.zip` hashes were unchanged at the time of that probe.

## Promoted conversion threads

`yuv_region_bgr` now calls `to_ndarray(format="bgr24", threads=1)`. Nothing else in that helper changed. The probe did not prove a swscale rebuild, and the code does not say that it did. `decode.py` SHA256 `9146d1d650b432c692d8484aa1dc922f0120a1dc35c56afa8b43e8f5c5cb0831`. `service.py` is still `e319791f323f1803a7d8dabb0e8d45eff2465de25aba7a0cb061bed5b6bd2d20`. `prepared-models.zip` is still `202a2d1808bcc49848739ca40fa8ca8b7c9f227bcbdb866476c592f3a4aee594`.

The full turbo suite was stopped. It was waiting on `cpu.lock` inside `test_s_package`, not computing. That stop is not a pass. The focused equality tests passed: 4 tests, 0.322 s, exit 0. Log `obj2/work/S/decode-threads-diag.txt`.

Confirmation, after that change, scratch cache, clip 03 phase 0: one warmup and three runs, no extra probes. Quiet pre-check passed on its last three samples (8.3%, 7.4%, 8.0%). Earlier samples in that window were not quiet, including 62.5%. All four documents hash `03c6a90e01989b7bcd0611622309c773a398a59515cd861a31a75843f98b72fd`. Warmup wall 2.931 s. The three runs are 2.715, 2.751, and 2.722 s. Process CPU 19.9 to 21.5 s. These are not H's final timing, and they are not a paired comparison with the old 3.59 s clock or with the 2.814 s probe median. Work counts stayed 563 decoded, 156 retrieved, 413 crops. The 156 figure is crop groups. Native pose requests are still one per crop.

Stage sums on those three runs still overlap: retrieve about 0.23–0.27 s, pose about 2.2 s, detect about 2.2 s, decode about 2.1 s. Those sums are not the wait on the last group.

## Separate timeline

One later run had wrappers on. Its 3.173 s wall is not a benchmark. The document still hashed `03c6a90e…`. Result `obj2/work/S/probe-promote-census.json`. Picture `obj2/work/S/promote-timeline.svg`.

The last finish is a tie. Group 4's player thread ends at 2.743 s. Group 2's ends at 2.739 s. Group 4 waited 1.691 s in `queue.get` for its first frame, from 0.443 s to 2.134 s. Reader A was busy with group 0, then group 2, then group 4. That wait is real. It is not empty time, and it is not the whole reason the preview is late: group 2 was already finding players for 1.618 s and finished 4 ms earlier.

Inside group 2's 1.618 s, detector calls took 0.716 s of wall and pose waves took 0.798 s. Caller CPU during those waits is about zero, so that time is not Python work. 14.95 s of process CPU sat on 43 threads that were already alive. That bucket is not split between the detector and the pose model. The tensor-copy lock could not be timed on its own.

After the last player thread, fill-in takes 0.269 s. Label takes 0.056 s. Writing the file takes 0.063 s. The join itself is under 1 ms.

OpenCV reported 16 threads. It was not changed. No next lever was started. One second and quality stay NOT MET.

## Readers, two versus four

Process-local only. Crop conversion stayed at one thread. Sampler, pose slots, pose threads, and models stayed put. The older single three-context 3.67 s row is not this pair. Production was not edited.

Quiet check before the runs: 6.5%, 7.6%, 8.8%. The clip has five groups. Result `obj2/work/S/probe-gop-parallel.json`.

Every warmup and every pair wrote the same document, hash `03c6a90e01989b7bcd0611622309c773a398a59515cd861a31a75843f98b72fd`. Work stayed 1080 frames, 563 decoded, 156 retrieved, 156 pose groups, 413 crops, 413 native pose calls, and 37 detector calls. The 413 figure is the native calls, counted apart from the 156 groups.

Two readers: 2.649 to 2.727 s, median 2.684 s. Four readers: 2.487 to 2.871 s, median 2.525 s. Pair gaps, two minus four: 0.196, 0.147, 0.163, 0.175, and −0.187 s. Median gap 0.163 s. The ranges overlap. Four of five pairs were lower with four readers. The last four-reader run was slower than every two-reader run. That pair also had the highest system CPU samples in the set, 68.5% and 71.3%, and those samples include this process. That does not prove an outside cause.

These walls include the same call counter and actor list on both sides. They are not the uninstrumented confirmation, and the old 1.691 s frame wait was not subtracted. One second and quality stay NOT MET. Same document bytes, so the stored miss still applies and was not re-scored.

## Fill-in, saved rows only

The guess was that the anchor scan inside `fill_frames` is the 0.269 s after the last player thread. It is not. No production edit. `sampler.py` is still `ade4caff9d293ed8b154db4847697ac70ffeefa78eafc4beecb46560333047dc`. Two readers stay.

Input is the rounded players from `gop-parallel/docs/gop2-pair-0.json`, hash `03c6a90e…`. Those rows are not the live unrounded tuples. No inference. Result `obj2/work/S/probe-fill-profile.json`. Picture `obj2/work/S/fill-split.svg`.

The helper itself, five warmed calls: 0.195 to 0.207 s, median 0.205 s. Caller CPU is 0.203 s on every repeat because that clock moves in 0.016 s steps. Caller CPU close to wall, within that grain, rules out a large wait on this reconstructed input. It does not rule out waiting inside the live 0.269 s call. The helper timer drops the returned frame lists before the second clock read, so that time includes throwing them away.

Inside one copied loop, 0.213 s: joint blend 0.099 s, building each row 0.083 s, the scan 0.019 s, box blend 0.007 s, the loop 0.004 s. The scan is 6,468 calls and 180,800 anchor visits, and still 8.8% of the helper. The cursor idea is closed. It was not timed, and there is no change. The failed cProfile record stays. The scan's 0.019 s is not a measured saving and not a preview saving.

Re-filling the rounded rows matches all 413 measured players. All 2,397 guessed players differ, by at most 0.01, one rounding step. None are missing. The live 0.269 s call was not rerun.

One second and quality stay NOT MET. The stored miss was not re-scored.

## Native async, same helper rules

Process-local only. The current thread helper stayed in production. Two readers and one conversion thread stayed. Pose slots, pose threads, sampler, and models stayed. The old three-pair 1.29 s spread is not this pair. No production edit.

Failure checks passed before the lock: every started request is waited, the real error is raised, and a failure does not start the next wave. Saved crops from one untimed clip03 phase0 capture, 156 waves and 413 crops, matched the thread helper exactly. That capture is not one of the twelve timed documents.

Quiet check waited through 12.9% and 12.5%, then 6.3%, 4.5%, 5.2%. Those samples include this process. Result `obj2/work/S/probe-infer-async-current.json`. Picture `obj2/work/S/infer-async-current.svg`.

Every warmup and every pair wrote the same document, hash `03c6a90e01989b7bcd0611622309c773a398a59515cd861a31a75843f98b72fd`. Work stayed 1080 frames, 563 decoded, 156 retrieved, 156 pose groups, 413 crops, 413 native pose calls, and 37 detector calls. The thread side counted 413 `ovcache.infer` calls. The async side counted 413 `start_async` calls, zero `ovcache.infer` calls, and request width 2 on all 156 groups. No run failed.

Current helper: 2.740 to 3.159 s, median 2.840 s. Async: 2.687 to 2.865 s, median 2.722 s. Pair gaps, current minus async: 0.053, 0.058, 0.101, 0.294, and 0.118 s. Median gap 0.101 s. All five pairs, including the two that ran async first, were lower on async. The ranges still overlap. Pair 3's current run was 3.159 s and had the highest system CPU sample, 75.6%, which includes this process. That pair is kept. It is not assigned to outside load.

Both medians are still near 2.7 s. This is not an adoption and not a path to one second. Quality stays NOT MET. The stored miss was not re-scored. Lock released.

One fake case was not in those checks. Request 0 starts and its wait fails. Request 1's start fails. A third crop is the next wave. No inference. Result `obj2/work/S/probe-infer-async-start-error.json`.

The current helper raises `wait failed`. The prototype waits request 0, drops that wait error, and raises `start failed`. Neither starts the third crop. Request 1 is not waited, because its start did not return. The lowest-index claim does not hold across a start failure. Production was not changed.
