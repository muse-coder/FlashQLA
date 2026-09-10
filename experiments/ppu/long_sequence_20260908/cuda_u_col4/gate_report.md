# Actual case3 seed1701 gate diagnostic

Actual PPU input:128K Hq2/Hv8, blog75, FP32 g/state, nonzero state; input g raw-bit SHA `3ac3d3b4ecddae36db5b1f6673d6d5881772b046220ef1c7b3ab54e61ff00eba`. This diagnostic copied the existing g to CPU, preserved its bits, and used float64 sums outside all timing. It did not regenerate inputs, alter RNG order, change kernels, or run AutoCP. The64chunk partitions are hypothetical; this is not bit-identical validation of main’s float32 GPU warmup kernel or a claim that AutoCP chose that partition size.

The actual six decaying heads are strong enough to cross cumulative log-gate<-10 in4–5chunks (256–320tokens). Every one of their31 hypothetical preprocessing partitions crosses the threshold with zero fallback. Heads6/7 are exactly zero and never cross; all31 corresponding partitions need the full64chunk fallback. The last32nd partition is ht_mask=True and requires0 preprocessing chunks; its fallback is marked not-required/undefined rather than falsely assigning a value. Output chunks are never skipped by this diagnostic.

|Head|g mean|min|max|Chunk64 sum median|Whole-tail chunks to<-10|31-partition warmup min/median/max|Fallback count|
|---|---:|---:|---:|---:|---:|---|---:|
|0|-0.040316327|-0.216629103|-0.000463367|-2.576090710|5|4/4/5|0|
|1|-0.040350141|-0.219696924|-0.000678566|-2.573607321|4|4/4/5|0|
|2|-0.040352688|-0.208034322|-0.000482166|-2.576613517|4|4/4/5|0|
|3|-0.040324880|-0.214590475|-0.000524575|-2.579275158|4|4/4/5|0|
|4|-0.040304419|-0.208429247|-0.000687520|-2.571933397|4|4/4/5|0|
|5|-0.040268926|-0.215671062|-0.000788448|-2.571981017|5|4/4/5|0|
|6|0.000000000|0.000000000|0.000000000|0.000000000|None|64/64/64|31|
|7|0.000000000|0.000000000|0.000000000|0.000000000|None|64/64/64|31|

Full min/p10/median/p90/max chunk-sum quantiles and every partition/head warmup result are in remote/gate_diagnostic.json. Exact input g FP32 words and chunk sums are in remote/raw_gate_seed1701.npz. Current candidate and7968 do not implement AutoCP, so these statistics cannot be cited as a measured AutoCP hit rate or AutoCP performance result.
