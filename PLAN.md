# PLAN.md — AMD MI210 (HIP/ROCm) Support for Mitsuba 3 + Dr.Jit

**Target hardware:** AMD Instinct MI210 — CDNA2, `gfx90a`, wavefront 64, ROCm.
**Consumer:** the fishsense pipeline running **Mitsuba 3 in a spectral variant** on MI210.
**Deliverable:** a working `hip_ad_spectral` (and `hip_ad_rgb`) Mitsuba variant.

**Status:** Phase 0a complete. **Phase 2 codegen complete except for the ray-tracing
opcodes: every Dr.Jit-Core test suite in the tree now passes against HIP** —
arithmetic, casts, memory, control flow, atomics, reductions, dynamic dispatch,
frozen-function recording and per-lane arrays. Dispatch is what Mitsuba plugin
selection rests on, so nothing structural now blocks a render. Phase 1 is now WRITTEN: dlopen bindings checked against a real ROCm install, device
enumeration, and a HIPThreadState whose every call shape is verified by executing the
same sequences through HIP-on-CUDA.

The Phase 0b spike is done. HIP-RT traversal links for gfx90a at a measured
64 VGPR / 38 AGPR / 800 B scratch, and it is also EXECUTED and verified correct on the
local NVIDIA GPU — an unusually good proxy, because gfx90a has no ray-tracing hardware
and so takes the same RTIP 0 software path. The spike turned up three findings that
change §7 (BACKEND_NOTES §7a-b).

What genuinely has no local proxy left: wave64 semantics, fp16 numerics, and the AMD
runtime's actual behaviour — the HIP calls are shape-checked through HIP-on-CUDA, but
nothing AMD-specific about them is.

All work so far done on an x86_64 NixOS laptop with an RTX 3060 and **no AMD
hardware** -- §0.2 covers why that is possible, §0.3 the CUDA shim, and §3.5
HIP-on-CUDA, which runs the real HIP API locally. What genuinely still needs the
MI210 is AMD-specific behaviour: wave64 semantics, fp16 numerics, and HIP-RT (§10).

This plan covers the **whole effort across all three layers**: Dr.Jit-Core (the bulk),
Dr.Jit (thin), and Mitsuba 3 (small). Splitting it into separate per-repo plans hides
the sequencing dependencies and — as shown in §2 — badly misestimates the Mitsuba side.

---

## 0. Timeline, and why HIP starts now

**Hardware availability decides the sequencing:**

| Resource | Available |
|---|---|
| MI210s | **Now** (owned) |
| Pixel fleet (thousands, bare-metal Debian/K8s) | **~6 months out** (soft estimate) |

**Decision: start the HIP port now. The MI210 target is settled** — §3.0 and §3.0.1 record
why, and are retained as rationale, not as open questions. Three reasons, none of them
technological:

1. **The MI210 is the only target testable for six months.** A Vulkan-first plan means
   half a year of blind development against absent hardware, with PanVK assumptions
   unvalidated until the fleet lands. Worst possible project shape.
2. **HIP delivers inside the window.** 4–7 months (§6) against a ~6-month horizon means
   working MI210 capability roughly when the phones arrive — versus nothing on either
   platform.
3. **The risk is asymmetric.** A thousand-phone procurement can slip. If it does, this
   looks better; if it arrives early, nothing is lost.

**The aarch64 CPU path is not a competing option.** `llvm_ad_spectral` builds on aarch64
today with **zero engineering** (verified below). What remains is orchestration — K8s Job,
seed splitting, EXR averaging, ~200 lines — which is a week's work *whenever* the fleet
lands. It never contended for these six months. It would only obviate the MI210 work by
making the cards unnecessary, which cannot be established for six months, during which the
cards sit idle. Treat it as a free complement, scheduled on hardware arrival.

**What makes HIP-now non-wasteful:** the §3.1 emit abstraction. Opcode lowering is the
expensive intellectual work and is largely target-independent, so it transfers to a SPIR-V
emitter later. **This elevates §3.1 from good practice to the load-bearing architectural
decision in this plan.**

**Run in parallel now — neither needs the fleet:**
- **Answer the three PanVK gates (§3.0.1),** `VK_KHR_buffer_device_address` above all.
  Answerable from existing PanVK work; makes the month-6 decision informed rather than
  guessed.
- **Measure a single phone if any unit exists.** Per-core aarch64 throughput extrapolates
  to the fleet. A day's work, and it sizes the whole question early.

**Month-6 decision point:** when the fleet lands, measure it against the then-working HIP
backend and decide whether to build the SPIR-V emitter on top of the (by then factored)
lowering layer. Re-read §3.0.1 at that point, not before.

### 0.1 Reference — the phone-fleet case, for the month-6 decision

**Why it is a real contender, not a joke:**
- `llvm_ad_spectral` **already runs on aarch64 today, with zero backend work.** Verified
  in this tree: drjit-core's LLVM backend has explicit aarch64/NEON handling
  (`llvm_eval.cpp`, `llvm_core.cpp`, `llvm_api.h`) and the vendored Embree has 29
  aarch64/NEON references. Apple Silicon support is why this path is maintained and
  exercised.
- **MI210 has no RT hardware** (§7.3), so HIP-RT does software BVH traversal. Embree on
  ARM is a first-class, heavily tuned path. Thousands of Embree-on-ARM cores are
  plausibly competitive with one MI210 at software traversal — the GPU's paper FLOPS do
  not apply to ray tracing here.
- Bare metal (no virtualization) means full 8–12 GB RAM and all cores per node, so scene
  fitting is largely a non-issue, and the governor can be tuned for sustained throughput.
- Rough capacity: ~5 effective cores/phone (Tensor cores are heterogeneous: 1–2 big,
  3–4 mid, 4 little) × 30–50% sustained after thermals ≈ **3,000–5,000 effective
  sustained ARM cores** at 4-wide NEON for a 2,000-phone fleet.

**The decisive question — what shape is the parallelism?**
- **Across many independent tasks** (a corpus of images, each its own reconstruction):
  one phone runs one whole problem, perfect linear scaling, *zero* coordination. Mitsuba
  3 has no built-in network rendering, and this needs none. **The MI210 port is close to
  pointless in this case** — its only edge (fast in-loop gradient iteration on a single
  problem) is irrelevant if no single problem is the bottleneck.
- **Within one render** (one large scene that must be fast): sample-parallel Monte Carlo
  still distributes cleanly — average independent-seed renders of N spp for exactly the
  k·N spp estimator, ~200 lines of K8s Job + object storage. But inverse rendering needs
  a cross-node gradient reduction *per iteration*, and network latency inside an
  optimization loop over hundreds of iterations is a cost a single MI210 does not pay.
  **This is the case where the GPU keeps its edge.**

**The measurement (run on any single aarch64 unit that exists):** build
`llvm_ad_spectral` for aarch64, run a representative fishsense workload, measure spp/sec.
Multiply by fleet size × throttle factor. Compare against the real MI210 numbers available
by then — accounting for **software** BVH traversal, not spec-sheet TFLOPS. This informs
the month-6 decision; it no longer gates project start, because the fleet does not exist
yet to be measured or deployed to.

**The Mali GPUs are a live option, not a ruled-out one.** On bare metal they are reachable
via Panfrost/Panthor (PanVK for Vulkan, rusticl for OpenCL), and **we have in-house PanVK
patches** — so driver-maturity gaps are fixable by us rather than blockers. Two objections
I initially raised have since failed: Mali's lack of FP64 is not disqualifying (drjit-core's
Metal backend already promotes `Float64`→`Float32`, since Apple GPUs lack it too), and RT
hardware is absent on the MI210 as well, so both targets do software traversal. This makes
a **Vulkan/SPIR-V backend a serious contender that would cover the fleet *and* the
MI210s** — see §3.0.1.

---

## 0.2 Spike results — measured, not assumed

Run locally on an x86_64 NixOS laptop (RTX 3060 Mobile, **no AMD hardware**). These
replace assumptions elsewhere in this document; each is cross-referenced from the
section it affects.

**§8.1 ANSWERED — HIP-RT supports `gfx90a`.** The project's #1 risk, resolved favorably.
`rocmPackages.hiprt` 3.0.3 is packaged and unbroken, and `hiprt_common.h` handles
`__gfx90a__` explicitly. **Phase 0b's gate is materially de-risked before it starts.**

**Software traversal confirmed (§7.3 caveat holds).** HIP-RT's RTIP tiers are 3.1
(gfx1200/1201) → 2.0 (gfx1100–1153) → 1.1 (gfx1030–1036) → **`#else #define HIPRT_RTIP 0`,
where `gfx90a` lands.** No hardware RT, exactly as predicted. The *availability* risk is
gone; the *performance* caveat is unchanged and still needs Phase 8 measurement.

**The full ROCm toolchain runs here without AMD hardware.** `hipcc` + `rocm-device-libs`
+ `clr` + `hiprt` all come from nixpkgs. HIP source compiles to a verified target:
```
bundle target:  hipv4-amdgcn-amd-amdhsa--gfx90a
ELF:            EM_AMDGPU, flags 0x53f, gfx90a, xnack, sramecc
ISA:            s_and_saveexec_b64, 64-bit vcc   (wave64 visible in the encoding)
```
This proves §7.1's claim: codegen errors are catchable without a GPU. **Phase 2 develops
against the real target**, which is better than the §3.5 NVIDIA fallback anticipated.

**§3.3 independently validated.** HIP-RT itself does:
```c
#if __gfx900__ || ... || __gfx90a__ || ... || __gfx942__
constexpr uint32_t WarpSize = 64;
#else
constexpr uint32_t WarpSize = 32;
#endif
```
An arch-varying compile-time constant — exactly the single-definition parameter §3.3
argues for. Follow HIP-RT's convention rather than inventing one.

**§7.4 quantified — compile latency is a non-issue.** Runtime source generation → NVRTC →
`cuModuleLoadData` → launch, verified working. Cold-compile scaling:

| src lines | PTX lines | compile |
|---|---|---|
| 106 | 260 | 24 ms |
| **1006** | **2060** | **76 ms** |
| 2006 | 4060 | 142 ms |
| 4006 | 8060 | 382 ms |

The real Mitsuba raygen megakernel is **2039 PTX lines**, so a megakernel-scale cold
compile is **~76 ms** — noise against a render, and removed entirely by the `~/.drjit`
cache. Caveats: `hiprtc` is not NVRTC, and the synthetic kernel is straight-line `fma`
with no calls or control flow, so treat this as an optimistic lower bound of the right
order. Note the superlinearity past ~2000 ops.

**Codegen scale measured.** The `cuda_ad_rgb` Cornell-box raygen kernel at 1 spp — the
*simplest possible scene* — is **2039 lines of PTX across 54 distinct opcodes**. That is
the floor for what the HIP emitter must cover.

**§3.3 evidenced in the hot path.** That same kernel contains `24 shfl`, `2 vote`,
`1 activemask`, `1 match`, `popc`, `brev`, `bfind`. Warp-level primitives are in **every
render**, not just the reduction library — which is why §7.2 is labelled wide blast radius.

**§3.2 validated emphatically.** The OptiX call in the real kernel is
`_optix_hitobject_traverse` with **~50 arguments** (32 outputs, 30 payload slots), against
`jit_metal_ray_trace`'s **8 outputs**. Mirroring Metal avoids roughly an order of magnitude
of interface complexity.

**Driver-ABI gotcha for the `hip_api.cpp` port.** `cuMemAlloc` / `cuMemcpyDtoH` /
`cuCtxCreate` must be bound as **`_v2`** symbols; the unversioned ones resolve fine but are
legacy variants with different signatures, failing at runtime with a misleading
`CUDA_ERROR_INVALID_CONTEXT`. drjit-core already handles this
([cuda_api.cpp:106](ext/drjit/ext/drjit-core/src/cuda_api.cpp#L106)); HIP versions symbols
the same way, so mirror it rather than rediscovering it.

---

## 0.3 Interim plan — developing on the local NVIDIA GPU

An MI210 host is being procured. Until it lands, **the RTX 3060 laptop is the development
platform**. This is more capable than §3.5 originally assumed, and a large fraction of the
project is unblocked.

### The dual-validation harness

Emit the kernel source **once**, then check it two independent ways. This is the core of
the interim setup and it exists because §3.1 chose source emission:

```
                      hip_eval.cpp emits HIP C++ source
                                    |
              +---------------------+---------------------+
              |                                           |
   hipcc --offload-arch=gfx90a                 NVRTC + CUDA driver API
   --genco  (compile only)                     (compile AND execute)
              |                                           |
   Does it compile for the REAL                Are the NUMBERS right?
   target? Address spaces, wave64              Compare against llvm_ad_rgb,
   intrinsics, ISA validity.                   which is already green here.
```

Neither path needs AMD hardware. Together they cover the two ways codegen fails: *invalid
for the target* and *valid but wrong*. Both were verified working in §0.2.

### What is unblocked, and what is not

| Phase | Status without an MI210 |
|---|---|
| 0a — wiring skeleton | **Full speed.** Never needed hardware. |
| 2 — codegen (6–10 wks, the big one) | **Full speed** via dual validation. |
| 3 — device library | **Full speed.** `hipcc` compiles `kernels.cu` for `gfx90a`; numerics check via the NVIDIA path. |
| 4 — advanced ops | **Mostly.** Software textures (§4) and scatter are testable; wave64 ballot **semantics** are not (see below). |
| 6 — Dr.Jit layer | **Mostly.** Traits and Python namespaces need no device. |
| 1 — runtime/driver | **Write, cannot execute.** Mechanical ~1:1 port of `cuda_api.cpp`, so low risk deferred. |
| 5 — ray tracing | **Blocked.** HIP-RT traversal must run. |
| 7, 8 — Mitsuba + spectral | **Blocked** end to end. |

That is roughly **Phases 0a, 2, 3 — the largest single block of work in §6** — proceeding
at full speed. The MI210 gates integration and execution, not the expensive intellectual
work.

### Use the existing CUDA runtime as a test harness

Do **not** wait on the HIP runtime layer to start executing generated kernels. drjit-core
already has a working CUDA backend built on the driver API. For interim testing, pair
**HIP codegen with the existing CUDA runtime path**: emit HIP C++ source, compile it with
NVRTC, and launch it through the machinery in `cuda_core.cpp` / `cuda_ts.cpp`.

This validates the expensive, high-variance part (codegen) against real execution while
deferring the cheap, mechanical part (the `cu*` → `hip*` runtime port), which carries
little risk and is trivially verified once hardware exists.

### The honest gap: wave64 semantics

The 3060 is warp-32. §3.3's width parameter means reduction and ballot paths are
*exercised* at 32, which catches structural errors — but **width-64 behaviour is not
verified**. Ballots do not fit a `b32`, and `shfl` lane masks differ. §0.2 measured 24
`shfl` + `vote` + `activemask` in a Cornell box at 1 spp, so this is the hot path of every
render, not a corner case.

Treat every wave64-dependent path as **unverified until MI210 access**, and keep a running
list of them rather than discovering it at integration time. This is §7.2, and it is now
the project's highest risk following §7.3's resolution.

### Optional: package HIP-for-NVIDIA

Executing against the *actual HIP API* locally would additionally unblock Phase 1. It
needs a `clr` override built with `__HIP_PLATFORM_NVIDIA__`, or vendored `nvidia_detail`
headers (§3.5). Feasibility is unconfirmed — the headers are absent from every nixpkgs
ROCm output *and* from `hip-common.src`, so this needs investigation before it is promised.
Not on the critical path: the dual-validation harness above covers codegen without it.

---

## 1. The most important fact about this tree

**This is not stock Mitsuba 3 / Dr.Jit.** A second GPU backend (Apple Metal) has
already landed all the way through the stack. That changes the character of the job:
we are *following a precedent*, not *inventing an abstraction*.

Concretely, every hard problem we face has already been solved once for a
non-NVIDIA GPU, in this tree:

| Problem | Existing non-CUDA answer to copy |
|---|---|
| New `JitBackend` woven into 26k LOC of shared core | `JitBackendMetal` |
| GPU codegen that isn't PTX | [metal_eval.cpp](ext/drjit/ext/drjit-core/src/metal_eval.cpp) — emits MSL **source text** |
| Precompiled device-library for a non-CUDA GPU | `resources/metal_kernels.metal` → `.metallib`, embedded |
| **Ray tracing without OptiX** | `jit_metal_ray_trace` — [metal.h:145](ext/drjit/ext/drjit-core/include/drjit-core/metal.h#L145) |
| Custom-shape intersection without OptiX IS programs | `MetalScene::intersection_fns` + per-scene intersection function tables |
| Mitsuba accel abstraction | [accel.h](include/mitsuba/render/accel.h), [scene_metal.inl](src/render/scene_metal.inl) |
| Backend-neutral geometry description | [scene_ir.h](include/mitsuba/render/scene_ir.h) |
| Python backend namespace | `drjit.metal` — `ext/drjit/src/python/metal.cpp`, `metal_ad.cpp` |

**Rule for this project: when in doubt, go read what Metal did.** Not what OptiX did.

### 1.1 Verified baseline

Measured in this checkout, not estimated:

| Component | LOC |
|---|---|
| `ext/drjit/ext/drjit-core/src/llvm_*` | 7,760 |
| `ext/drjit/ext/drjit-core/src/cuda_*` | 6,668 |
| `ext/drjit/ext/drjit-core/src/metal_*` | 6,505 |
| `ext/drjit/ext/drjit-core/src/optix_*` | 1,922 |
| drjit-core shared core | 25,992 |
| `cuda_eval.cpp` / `metal_eval.cpp` (codegen) | 1,900 / 1,675 |
| Mitsuba OptiX accel (`scene_optix.inl` + `optix/accel.cpp`) | 818 + 403 |
| Mitsuba Metal accel (`scene_metal.inl` + `metal_accel.mm`) | 175 + 694 |

`ThreadState` ([internal.h:870](ext/drjit/ext/drjit-core/src/internal.h#L870)) declares
**19 pure-virtual** methods (27 virtual total); `ThreadStateBase` adds 1. That interface
is our implementation checklist.

Backend-predicate sites in drjit-core: `jitc_is_gpu` 18, `jitc_is_cuda` 54,
`jitc_is_metal` 46, `jitc_is_llvm` 33. The 18 `jitc_is_gpu` sites we inherit for free.

---

## 2. Scope split across the three repos

### Layer A — Dr.Jit-Core (`ext/drjit/ext/drjit-core`) — ~90% of the work
A new `hip_*` backend family. Budget **~5–7k LOC** against the Metal yardstick.

### Layer B — Dr.Jit (`ext/drjit`) — small, mechanical
A `drjit.hip` / `drjit.hip.ad` Python namespace mirroring `src/python/metal.cpp` and
`metal_ad.cpp`, plus `is_hip_v` traits threaded through `array_traits.h`, `jit.h`,
`autodiff.h`, `tensor.h`, `texture.h`, `dlpack.cpp`, `meta.cpp`, `base.cpp`,
`freeze.cpp`. **~500–1,500 LOC**, mostly copy-adapt.

### Layer C — Mitsuba 3 (this repo) — small, and much smaller than it looks
- `include/mitsuba/render/accel_hip.h` + `src/render/scene_hip.inl` — modelled on the
  **Metal** pair (175 + 694 L), not the OptiX pair (818 + 403 L).
- `SceneAccel` trait branch in [accel.h](include/mitsuba/render/accel.h) (47-line file).
- Variant registration: `MI_ENABLE_HIP` block in
  [CMakeLists.txt:173-190](CMakeLists.txt#L173-L190) and the prefix checks in
  [resources/configure.py:26](resources/configure.py#L26),
  [:75](resources/configure.py#L75), [:188-190](resources/configure.py#L188-L190).
- ~20 files with `MI_ENABLE_CUDA` / `is_cuda_v` guards to generalize — concentrated in
  [ellipsoids.cpp](src/shapes/ellipsoids.cpp) (8), [sdfgrid.cpp](src/shapes/sdfgrid.cpp) (6),
  [shapegroup.cpp](src/render/shapegroup.cpp) (6), and the custom-shape AABB paths in
  [sphere.cpp](src/shapes/sphere.cpp) / [cylinder.cpp](src/shapes/cylinder.cpp) /
  [disk.cpp](src/shapes/disk.cpp).

**Critically:** [scene_ir.h](include/mitsuba/render/scene_ir.h) is already
*"a backend-neutral IR to feed acceleration structure builders"*, with a `Kind` enum
covering Triangles / TrianglesCulled / BSplineCurve / LinearCurve / Custom / Instance
and `fill_aabbs` / `fill_data` callbacks. `SceneIRBuilder` already emits exactly the
geometry description a HIP BVH builder consumes. **This is the single biggest reason
Layer C is cheap.**

### Explicitly out of scope
- **Cooperative vectors / MFMA / rocWMMA.** Mitsuba does not use `coop_vec` at all
  (grep finds it only in release notes). Stub the interface; skip the implementation.
- **Green contexts** (`cuda_green.cpp`) — no HIP analog.
- **OptiX denoiser** ([optixdenoiser.cpp](src/render/optixdenoiser.cpp), 535 L with
  header) — no AMD analog. Compile it out of HIP variants.
- **Multi-generation AMD support.** `gfx90a` only. No RDNA, no support matrix.

*(HIP-on-NVIDIA is emphatically **in** scope as a test platform — see §3.5. It would
have been useless had we emitted AMDGPU IR, but source emission makes it valuable.)*

---

## 3. Key architectural decisions

### 3.0 Why HIP, and not OpenCL / Vulkan / SYCL / WebGPU

Recorded because it is the first question any reviewer asks, and because the answer is
driven less by language preference than by two hard requirements: **ray tracing with
custom intersection callbacks**, and **the specific card being an MI210**.

| Option | Ray tracing | FP64 | MI210 viability | Verdict |
|---|---|---|---|---|
| **HIP** | HIP-RT, custom callbacks | Full-rate, first-class | Primary ROCm target | **Chosen — available now (§0)** |
| OpenCL | None credible | Extension-gated, patchy | Second-class on ROCm | Rejected |
| **Vulkan** | Software BVH — *same as HIP* | Optional; Metal proves it | Serves phones **+** MI210s | **Deferred to month 6 (§3.0.1)** |
| SYCL | Interop fight, 2 compilers | First-class | Via HIP, but model mismatch | Rejected (closest call) |
| WebGPU | None | **None at all** | N/A | Not a candidate |

**OpenCL** has no credible ray-tracing story — Radeon Rays is effectively superseded by
HIP-RT — and AMD's OpenCL is a second-class citizen on ROCm, where compute investment,
`rocprof`, and `rocgdb` all center on HIP. FP64 atomics are extension-gated. The
empirical signal is decisive: **Blender dropped OpenCL from Cycles in 3.0 and moved to
HIP for AMD** — the same problem shape as ours (production GPU path tracer needing AMD
support). Also: drjit-core has no OpenCL precedent to follow.

**Vulkan — ⚠ REOPENED. See §3.0.1.** The original rejection has not survived new
information (an existing bare-metal Mali fleet plus in-house PanVK patches) and is
retained below only to show what changed:

> ~~It fails on the hardware, not the design: **MI210 is a compute-only datacenter part**,
> Instinct support runs through ROCm rather than the graphics stack, and Vulkan
> ray-tracing extensions on `gfx90a` are unreliable-to-absent. Secondary strikes:
> `shaderFloat64` is optional and commonly slow, wasting MI210's full-rate FP64; no MFMA
> access; and descriptor sets / pipeline layouts / barriers / command buffers are enormous
> boilerplate next to `hipMalloc` + `hipModuleLaunchKernel`.~~

**Why each leg failed:**
- *"Vulkan RT unreliable on `gfx90a`"* — **irrelevant.** MI210 has no RT hardware either
  (§7.3), so traversal is software on both targets. Writing BVH traversal into the
  generated compute shader (§7.3 fallback 2) is needed for Mali regardless, and serves
  both. This was HIP's implicit advantage; it is not one.
- *"`shaderFloat64` optional/slow"* — **retired.** drjit-core already ships a GPU backend
  with **no FP64 at all**: [metal_eval.cpp:291](ext/drjit/ext/drjit-core/src/metal_eval.cpp#L291)
  promotes `Float64` → `Float32` with a warning, because Apple GPUs lack it. Mitsuba's
  spectral variants are float32. Not disqualifying.
- *"Enormous boilerplate"* — **still true, but repriced.** The objection assumed no
  in-house Vulkan expertise. That assumption is false.
- *"SPIR-V codegen is a lift"* — **overstated.** Emit GLSL compute source → shaderc →
  SPIR-V → `vkCreateShaderModule` is structurally identical to the Metal path (emit MSL
  source → runtime compile), so the §3.2 Metal template applies cleanly. The thick part
  is the Vulkan *runtime*, not the codegen.

### 3.0.1 Rationale of record: HIP vs. Vulkan/SPIR-V vs. aarch64 CPU
*(Settled — HIP. Retained as the reasoning behind the decision, not as an open question.
Revisit only at the §0 month-6 review.)*

| Target | Reach | Engineering | RT story |
|---|---|---|---|
| **aarch64 CPU** | Thousands of phones | **Zero — works today** | Embree, tuned, first-class |
| **Vulkan/SPIR-V** | Phones + MI210s + dev boxes | Large; new, thick runtime | **You own the BVH** |
| **HIP** | MI210s only | Large; runtime ports ~1:1 | HIP-RT, free |

**Strongest surviving HIP argument:** HIP-RT hands us a tuned BVH builder *and* software
traversal for free. Vulkan means writing both ourselves in a generated compute shader,
and it will likely be slower. That is real, deliberate scope being taken on.

**Strongest Vulkan argument:** one backend covers the fleet we already own, the MI210s we
already own, and local dev machines — and it drops the §8.1 dependency on HIP-RT
supporting `gfx90a`, currently the project's #1 risk.

**Three gates decide Vulkan.** All should be answerable from existing PanVK work:
1. **`VK_KHR_buffer_device_address` on PanVK — make-or-break.** Dr.Jit passes *raw device
   pointers* into kernels. Without BDA, every buffer needs a descriptor set, which changes
   the codegen model substantially.
2. **Subgroup ops** (`subgroupBallot`, shuffles, arithmetic) for reduction/prefix-sum
   paths. Mali subgroups are narrow and variable-width — another vote for §3.3's
   width-as-parameter over a hard constant.
3. **`VK_EXT_shader_atomic_float`** — scatter-add is load-bearing for Mitsuba's gradient
   accumulation.

Plus: **does RADV support `gfx90a` for compute?** If yes, one Vulkan backend genuinely
covers both fleets.

**Resolution: HIP now, on sequencing grounds — see §0.** The technical case for Vulkan is
genuinely strong and is *not* what decides this. What decides it is that the MI210s are
available today and the Mali fleet is ~6 months out, making Vulkan-first six months of
blind development. Revisit at the §0 month-6 decision point, with the three gates already
answered and the §3.1 lowering layer already factored.

**Note on sunk cost.** Owning the MI210s argues for using them, but the 4–7 months of
engineering is the scarce resource, not the cards. They remain available under any choice —
which is why availability *timing*, not ownership, is the deciding factor.

**SYCL / oneAPI** is the closest call after Vulkan and deserves more than a one-liner,
because the naive dismissal ("it lowers to HIP anyway") is actually backwards — lowering
to HIP means you *inherit* ROCm codegen quality without writing an AMD backend, which
argues for SYCL. Three things decide against it:

1. **Model mismatch.** SYCL is single-source and compile-time-oriented: you write C++
   kernel lambdas and the compiler splits host from device. Dr.Jit traces at runtime,
   generates kernel text at runtime, compiles at runtime, and launches. Every headline
   SYCL feature — single-source, buffers/accessors, managed dependency graphs, USM — lives
   in precisely the layer Dr.Jit *replaces*. What Dr.Jit needs is thin: allocate, memcpy,
   compile-this-string, load-module, launch. All three existing backends bind at exactly
   that level. HIP fits the shape; SYCL fights it.
2. **Ray-tracing interop.** HIP-RT's traversal is a device-side C++ template that must be
   `#include`d into the kernel. Reaching it from SYCL means backend interop
   (`sycl::get_native`) for native handles *plus* getting HIP-RT's device templates
   through SYCL's device-compilation pass — a fight with two compilers, not a seam.
   (Embree 4's SYCL GPU path does not rescue this: it requires an Intel GPU and Level
   Zero, and does not run on AMD.)
3. **Schedule risk on the critical path.** Runtime kernel compilation in SYCL rests on
   experimental extensions (`sycl_ext_oneapi_kernel_compiler`; AdaptiveCpp's SSCP JIT).
   Building this project's core mechanism on an experimental extension is a poor bet
   against a fixed card and a schedule.

**What the SYCL case gets right, and how we answer it.** Its strongest argument is
strategic, not tactical: drjit-core accumulates ~6,500 LOC of hand-written backend per
vendor (CUDA, Metal, now HIP), and one SYCL backend would cover AMD + NVIDIA + Intel +
CPU. That is a real cost we are accepting. But note that for a runtime-tracing JIT,
SYCL's *language* is useless and only its *runtime* is interesting — and if we emitted
SPIR-V we would want Level Zero directly, not SYCL's runtime. So "the SYCL option"
decomposes into two independent choices — **what we emit** and **what we launch
through** — and SYCL is not clearly the best answer to either.

That is where the portability actually lives: **in the emit abstraction, not the API
choice.** Opcode lowering (how `fma`, `select`, gather/scatter, and loop control flow map
to a target) is the expensive intellectual work and is largely target-independent. Keep it
factored behind the §3.1 emit layer and a future SPIR-V / Level Zero emitter reuses it —
buying the multi-vendor hedge inside our own architecture, without SYCL's model mismatch,
experimental-extension dependency, or RT interop problem. **This promotes the §3.1 emit
layer from an implementation nicety to an explicit architectural requirement**, now
justified twice over: source↔IR reversibility *and* multi-vendor optionality.

**Revisit trigger:** if the group's hardware roadmap turns toward Intel GPUs, reopen this.
That is where SYCL's story is strongest and where Embree-on-SYCL would actually be
available.

**WebGPU** has no FP64 at all, no ray tracing in the standard, and sandbox-shaped compute
limits. It is a portability layer for web content, not a datacenter compute API.

**The affirmative case for HIP:**
1. `gfx90a` is a **primary ROCm target** — comgr, `rocprof`, `rocgdb`, hipBLASLt all
   assume HIP.
2. **HIP-RT is the only credible OptiX analog** offering custom intersection callbacks,
   which Mitsuba's implicit shapes (sphere, disk, cylinder, sdfgrid, ellipsoids) require.
3. **Full-rate FP64 is reachable** — MI210's headline capability, and directly relevant to
   Mitsuba's double variants.
4. **Closest to the code we are porting.** CUDA→HIP is near-1:1 at the API level;
   `cuda_api.cpp` ports mechanically. Every other option is from scratch at both the API
   and codegen layer.
5. **It is the only option that is AMD-first *and* testable on NVIDIA** (§3.5). The
   vendor-neutral options are equally foreign to both platforms and buy nothing here.

**Honest caveat:** were the requirement "run on any GPU" rather than "run on MI210,"
Vulkan compute + `ray_query` would deserve serious study. That is not the requirement —
and the MI210 is the worst card in AMD's lineup on which to attempt it.

### 3.1 Codegen target: **HIP C++ source → `hiprtc`**

The three existing backends sit at different abstraction levels: CUDA emits **PTX
assembly text**, Metal emits **MSL source text**, LLVM emits **LLVM IR**. We emit
**HIP C++ source** and compile at runtime with `hiprtc`.

**Why not AMDGCN ISA text (the PTX-shaped choice):** PTX has *virtual registers* and
the driver performs register allocation and scheduling. AMDGCN ISA text uses *physical*
VGPRs/SGPRs — emitting it means writing our own register allocator. Disqualifying.

**Why not AMDGPU LLVM IR (the tempting choice):** it looks like the natural PTX analog,
and it does solve register allocation. But it **cannot `#include` anything** — and our
ray-tracing dependency (HIP-RT) is a *device-side C++ template library*
(`#include <hiprt/hiprt_device.h>`, then instantiate `hiprtGeomTraversalClosest`).
Custom intersection functions (`hiprtFuncTable`) must be compiled into the same module.
Note *why Metal gets away with inline RT*: it emits source text, so its kernels can
name `raytracing::intersector` and an `intersection_function_table` directly. RT is
what forces the abstraction level. Choosing IR would mean linking precompiled HIP-RT
bitcode through comgr — possible, but pioneering, and it still leaves custom
intersection functions unsolved.

**Why source wins — two load-bearing arguments, not a long list:**
1. **HIP-RT integration.** A `#include` versus unprecedented bitcode linking with
   hand-matched ABI, on the critical path, in the highest-variance phase. This is the
   argument with no good IR answer.
2. **Debuggability compounded over 4–7 months.** Dump the kernel, read it as C++, compile
   it standalone, run it under `rocgdb`, bisect — versus reading thousands of lines of
   numbered SSA. Daily, for the project's duration.

Secondary: it collapses Phase 3, since the device library compiles with the same toolchain
as the generated kernels.

**Two arguments deliberately *not* relied on**, because they were tested and did not hold:
- *"It is the Metal-shaped choice."* Metal emits MSL because Apple exposes nothing lower to
  third parties — a constraint, not a design choice. Citing a forced move as precedent for
  a free choice is invalid. What survives is weaker but still useful: Metal *demonstrates*
  that RT integration through source emission works, which is the specific property HIP-RT
  needs. Evidence, not precedent.
- *"Source has a smaller silent-miscompile surface."* Two-sided. AMDGPU IR has address
  spaces, calling conventions, and datalayout — learned once, encoded once. Generated C++
  has integer promotion, operator precedence, implicit conversions, and header name
  collisions. Roughly a draw; see the mitigation below.

**Mitigation — defensive emission (required).** Generated C++ does not need to be
idiomatic. **Emit fully-parenthesized expressions and explicit casts everywhere**, so
precedence and integer-promotion hazards are eliminated by construction rather than by
care. Prefix all generated identifiers to avoid collisions with HIP/HIP-RT headers.

**Known cost:** `hiprtc` runs a full Clang front-end per kernel, so compile latency is
materially worse than PTX→driver. Dr.Jit's disk kernel cache absorbs most of this in
steady state (§7.4).

**The emit abstraction is an architectural requirement, not a nicety.** Write the
per-opcode render templates against a thin emit layer rather than hard-coding `hiprtc`
assumptions. `cuda_eval.cpp`'s `fmt`-based template engine and `metal_eval.cpp`'s are
structurally the same; keep ours that way. This layer is justified twice over:

1. **Reversibility.** If `hiprtc` compile latency proves fatal (§7.4), switching to
   IR + comgr is a backend swap rather than a rewrite.
2. **Multi-vendor optionality.** Opcode lowering — how `fma`, `select`, gather/scatter,
   and loop control flow map to a target — is the expensive intellectual work and is
   largely target-independent. Factored properly, a future SPIR-V / Level Zero emitter
   reuses it. This is how we buy the portability that §3.0's SYCL case correctly asks
   for, without adopting SYCL's programming model.

### 3.2 Ray tracing: mirror `jit_metal_ray_trace`, HIP-RT underneath

**Do not design this against OptiX.** OptiX's pipeline + SBT + callable-program model
restructures the whole kernel and accounts for much of the 1,922 LOC `optix_*` layer
and Mitsuba's 818-line `scene_optix.inl`. That is not our problem shape.

`jit_metal_ray_trace` already defines the interface we want, verbatim:

```c
void jit_metal_ray_trace(uint32_t n_args, uint32_t *args, uint32_t mask,
                         uint32_t *out, uint32_t n_out, uint32_t scene, int shadow);
```

...with an 8-output hit record (hit flag, t, uv, instance/primitive/geometry/user IDs),
a `shadow` early-out flag, a scene handle passed as a JIT variable via
`jit_metal_configure_scene`, per-scene intersection-function-table caching, and a death
callback for scene lifetime. **`jit_hip_ray_trace` mirrors this; `HIPScene` mirrors
`MetalScene`.** The API design, the custom-shape mechanism, and lifetime handling are
all settled work. We swap HIP-RT in for `raytracing::intersector` underneath.

**MI210 has no ray-tracing hardware.** CDNA2 ships zero RT accelerators (those are
RDNA2+), so HIP-RT performs software BVH traversal. Correct but unaccelerated — an
accepted, measured-early caveat of this card (§7.3).

### 3.3 Wavefront width as a single-definition parameter (64 on `gfx90a`)

`gfx90a` is **wave64-only** — CDNA cannot do wave32 at all. Unlike RDNA, a fixed 64 is
genuinely safe on the target.

**But do not hardcode it inline.** Define the width in exactly one place and have
codegen read it. Reason: the NVIDIA test platform (§3.5) is warp-32, and cross-lane code
built for width 64 on a warp-32 device does not fail to compile — it silently computes
wrong results. That is worse than no coverage, because it manufactures false confidence
in precisely the area §7.2 flags as risky. With a single-definition parameter, the
NVIDIA build sets 32 and exercises the reduction/ballot paths for real; only genuinely
width-64-specific behavior then requires the MI210.

Confirmed sites needing work (the HIP counterparts of the first are **done** — see
`hip_ts.cpp`, where the four launch-configuration constants are read from `HIPDevice`,
and `resources/common.h`, where `WarpSize` is the parameter; the CUDA files below are
untouched and stay at 32, which is correct for them):
- [cuda_ts.cpp:800](ext/drjit/ext/drjit-core/src/cuda_ts.cpp#L800) — `const uint32_t warp_size = 32`, plus the 32-element grouping comments at :842 and :854.
- [cuda_scatter.cpp](ext/drjit/ext/drjit-core/src/cuda_scatter.cpp) — the peer-aggregation path built on `activemask.b32` (:63), `shfl.sync.bfly.b32` with 31-clamps (:125-131, :157-163, :332), and `vote.sync.ballot.b32` (:171). **A 64-wide ballot does not fit a `b32`** — this needs redesign against `__ballot64` / `__lane_id()`, not mechanical substitution.

### 3.4 Dynamic runtime loading
Follow the CUDA/Metal precedent: `DRJIT_ENABLE_HIP` + `DRJIT_DYNAMIC_HIP`, resolving
`libamdhip64.so` symbols at runtime so the library builds and loads without ROCm
present. [cuda_api.cpp](ext/drjit/ext/drjit-core/src/cuda_api.cpp) already does exactly
this and ports near 1:1 (`cu*` → `hip*`).

### 3.5 Local development without an MI210 (revised after §0.2 measurements)

> **Revision.** This section was written on the premise that we could not compile for AMD
> locally, making NVIDIA the necessary proxy. **That premise was wrong** (§0.2): `hipcc`
> produces verified `gfx90a` code objects on an ordinary x86_64 NixOS laptop with no AMD
> hardware. Codegen validation — the bulk of Phase 2 — needs no NVIDIA proxy at all.
>
> What NVIDIA would still add is **execution**: running a generated kernel and checking the
> numbers. That remains valuable, with one packaging caveat.
>
> **✅ HIP-for-NVIDIA WORKS. An earlier revision of this section said it did not —
> that was a bad search, not a real finding.** Verified end to end: `hipMalloc`,
> `hipLaunchKernelGGL` and `hipMemcpy` compiled by `nvcc` and executed on the local
> RTX 3060.
>
> The headers were not missing, they were **somewhere else**. AMD split the NVIDIA
> backend into its own repository, **`ROCm/hipother`**, under
> `hipnv/include/hip/nvidia_detail/`. Searching `ROCm/HIP` — where its own README says
> they live — turns up nothing, which is what produced the wrong conclusion.
>
> Four ingredients, none of which the error messages point at. Each was found by
> following the include chain one failure at a time:
>
> | Need | Package |
> |---|---|
> | `nvidia_detail/` headers | `ROCm/hipother` (flake input, **not** in nixpkgs) |
> | `cuda_runtime.h` | `cudaPackages.cuda_cudart` |
> | `cuda_profiler_api.h` | `cudaPackages.cuda_profiler_api` |
> | `nv/target` | `cudaPackages.cuda_cccl` |
>
> **The version must match.** `hipother` 7.2.0 against `clr` 7.2.3 works; 6.2.0 fails
> deep inside the AMD-side header on an undefined `hipHostAllocDefault`, which reads
> like a packaging fault rather than version skew.
>
> All of it is wired into `nix develop .#hip` as `$HIPNV_CFLAGS` / `$HIPNV_LDFLAGS`.
>
> **Why this matters more than the CUDA shim.** The shim (§0.3) runs HIP-*shaped*
> codegen on the CUDA runtime and never touches the HIP API. This runs the **actual
> HIP API**, so `hip_api.cpp` / `hip_ts.cpp` — Phase 1 — can be written and tested
> here rather than blind against hardware we do not have.
>
> **The limit, unchanged and worth stating plainly.** On NVIDIA, HIP is a header-level
> translation to CUDA. This validates **API usage** — signatures, argument order, flags,
> error handling — and says nothing about AMD behaviour. Wave width is still 32, and
> `hipMalloc`'s MI210 semantics remain unverified.
>
> | Capability | Status |
> |---|---|
> | Compile for real `gfx90a` | **Works** (§0.2) |
> | Execute generated kernels locally | **Works** — NVRTC-direct |
> | Execute against the actual HIP API locally | **Works** — HIP-on-CUDA |
> | Validate AMD-specific behaviour / wave64 | Still needs the MI210 |

#### Original rationale (still correct on the mechanics)

HIP is a portable language, not an AMD-only one. On an NVIDIA host, `hipcc` wraps
`nvcc`, the HIP headers inline `hip*` calls to `cuda*`, and — the part that matters —
**`hiprtc` maps to NVRTC**. So the exact flow we are building (generate HIP C++ source →
runtime-compile → load module → launch) runs on a consumer NVIDIA card.

**This is only true because §3.1 chose source emission.** Had we emitted AMDGPU LLVM IR,
an NVIDIA GPU could not execute it and this option would not exist. It is a direct
dividend of that decision.

**What it covers:** Phase 2 — the 6–10 week codegen block, the largest single piece of
labor in the project — plus Phases 3, 4, and 6 in large part. Local iteration with no
contention for MI210 access, and a viable CI target.

**What it does *not* cover:**
- **Width-64 cross-lane behavior.** See §3.3 — mitigated by the width parameter, not
  eliminated. Ballot/shuffle correctness at 64 needs the real card.
- **The driver shim (§3.4).** NVIDIA-platform HIP is header-inlined to CUDA; there is no
  `libamdhip64.so` to `dlopen`, so `DRJIT_DYNAMIC_HIP` is unexercised. Phase 1 needs the
  MI210.
- **AMD-specific intrinsics** — `__builtin_amdgcn_*`, `unsafeAtomicAdd`, DS ops. Keep
  these behind a narrow, clearly-marked set of emit helpers so the portable core stays
  portable and the non-portable surface stays visible.
- **Performance.** Meaningless for MI210. And FMA contraction, denormal handling, and
  transcendental accuracy all differ, so validate to tolerance — never bit-exact.
- **HIP-RT** — plausible via HIP-on-CUDA, but unconfirmed. See §8.6.

**Two ways to build it.** Either a genuine HIP-on-CUDA toolchain (higher fidelity on the
HIP API surface), or — simpler — compile the generated source with **NVRTC directly**
behind a small `#define` shim for the handful of device-side HIP-isms, and launch through
the `libcuda` path drjit-core **already has** in `cuda_api.cpp` / `cuda_core.cpp`. NVRTC
is not currently used anywhere in drjit-core (the CUDA backend loads only precompiled
PTX), so this is new machinery either way — but the NVRTC route is *less* new machinery
and needs no second dynamic-loading configuration. The HIP API surface is not what we are
testing; codegen is. **Start with NVRTC-direct.**

**The guard — write this on the wall.** NVIDIA-runnability is a **testing convenience,
never a design constraint.** The moment it conflicts with doing the right thing for
`gfx90a` — an MFMA path, an AMD atomic, a wave64-shaped reduction — MI210 wins and the
NVIDIA path loses coverage for that piece. Stated explicitly because this is exactly the
kind of thing that quietly decays into "we can't use that intrinsic, it would break CI."

---

## 4. Port vs. rewrite

| Area | Source | Effort | Notes |
|---|---|---|---|
| Driver shim | `cuda_api.cpp/.h` | **Port ~1:1** | HIP driver API mirrors CUDA. Drop green contexts. |
| Device/core setup | `cuda_core.cpp` | **Port** | Enumerate, module compile/cache, streams, events. |
| ThreadState | `cuda_ts.cpp` | **Port** + wave64 | Launch bookkeeping, memcpy, memset. |
| **Codegen** | `cuda_eval.cpp` (1,900 L) | **Rewrite** | PTX → HIP C++ source. Structure from `cuda_eval`; abstraction level from `metal_eval`. **The big item.** |
| Ray tracing | `metal_eval.cpp` :916-1000 + `metal.h` | **Adapt** | Mirror `jit_metal_ray_trace` / `MetalScene`. |
| Scatter/atomics | `cuda_scatter.cpp` | **Rewrite** | Different atomic/intrinsic set; wave64 ballots. |
| Packet memory | `cuda_packet.cpp` | **Rewrite** | |
| Textures | `cuda_tex.cpp` | **Rewrite, deferrable** | See §5 Phase 4. |
| Device library | `resources/kernels.cu` + `*.cuh` | **Recompile** | `hipcc` → `gfx90a` code object; wave64; matching embed/pack step. |
| Coop-vec / GEMM | `optix_coop_vec.cpp`, `cuda_ts.cpp` | **Stub** | Out of scope (§2). |

**Do not forget the second kernel set.** Reductions, prefix sums, `compress`, `mkperm`,
and `gemm` are *not* codegen'd. They are hand-written CUDA in `resources/`
(`block_reduce.cuh`, `block_prefix_reduce.cuh`, `reduce_2.cuh`, `compress.cuh`,
`mkperm.cuh`, `misc.cuh`, `gemm.cuh`), compiled by `resources/Makefile` to
`kernels_75.ptx`, LZ4-packed to `kernels_75.lz4` via `pack.c`, embedded, and loaded at
runtime. The Metal equivalent (`metal_kernels.metal` → `.metallib`) shows the pattern
for a non-CUDA target. We need a `hipcc` rule and a `gfx90a` code object with a
matching embed step.

---

## 5. Phased plan

### Phase 0 — Wiring skeleton + de-risking spike (parallel)

> **✅ 0a DONE.** `JitBackend::HIP = 4`, `Count = 5`, `jitc_is_hip()`, `HIPDevice`
> (carrying `warp_size` as §3.3's single-definition parameter), dispatch arms in
> `init.cpp` / `api.cpp` / `eval.cpp`, and `DRJIT_ENABLE_HIP` (default OFF) in CMake.
> `jitc_hip_init()` returns false, so the backend is registered but inert.
> Locked in by `tests/hip_wiring.cpp`; both build paths verified clean.
>
> **⏳ 0b BLOCKED on MI210.** Needs hardware by definition. Note §0.2 has already
> de-risked its main question: HIP-RT supports `gfx90a`.

**0a. Wiring (mechanical, drjit-core).** Add `JitBackendHIP = 4` and bump
`JitBackendCount` in
[jit.h:70-76](ext/drjit/ext/drjit-core/include/drjit-core/jit.h#L70-L76); extend
`jit_backend_name_v`. Add `jitc_is_hip()`. Add `HIPDevice` + `state.hip_devices`.
Thread through the dispatch files (`init.cpp`, `eval.cpp`, `op.cpp`, `api.cpp`,
`call.cpp`, `var.cpp`, `malloc.cpp`, `io.cpp`, `reorder.cpp`) — grep `jitc_is_metal`
for the exact 46-site set. Add `DRJIT_ENABLE_HIP` / `DRJIT_DYNAMIC_HIP` and
`DRJIT_HIP_FILES` to CMake, mirroring the Metal blocks.
*Milestone: builds with the backend registered but inert.*

**0b. The spike — this is the real gate.** Not "compile a trivial kernel." The spike
must be **a generated HIP C++ kernel that performs a HIP-RT traversal against a built
BVH, with one custom intersection callback**, compiled via `hiprtc` and launched
through the HIP driver API on real MI210 hardware.

Rationale: a trivial-kernel spike will succeed and teach us nothing. The question that
can sink this project is whether the RT integration composes with runtime-generated
source. If that shape works, the rest is labor. **Gate: 3 weeks.** If it does not
clear, take the §7.3 fallback before writing any codegen.

**Write the spike as `tests/hip_triangle.cpp`**, mirroring drjit-core's existing
`optix_triangle.cpp` / `metal_triangle.mm`. Same work either way, but it lands as a
permanent regression test instead of throwaway scaffolding (§5.9 gap 1).

### Phase 1 — Runtime + driver layer
Port `cuda_api` → `hip_api`, `cuda_core` → `hip_core`, minimal `hip_ts`.
`jit_init(JitBackendHIP)` enumerates the MI210; alloc/free; host↔device memcpy.
*Milestone: create, write, and read back a `HIPArray<float>`.*

### Phase 2 — Codegen (the big one)

> **⏳ IN PROGRESS — the emitter works and is verified against the upstream suites.**
>
> **Done — the emitter.** `jitc_hip_render()` and `jitc_hip_assemble()` cover
> arithmetic, comparisons, select, casts and bitcasts, transcendentals, bit
> counting, wide multiply, gather/scatter, scatter-reduce, the returning atomics
> (`ScatterInc` / `ScatterExch` / `ScatterCAS`), and control flow (symbolic loops
> and `if`).
>
> **Done — real coverage.** HIP is registered with the `TEST_*` macros in
> `tests/test.h`, so the upstream suites run against it:
>
> | suite | HIP | covers |
> |---|---|---|
> | `test_basics` | 7/7 | the full op matrix vs. constant folding, all types |
> | `test_loop` | 9/9 | symbolic loops — **control flow is verified** |
> | `test_mem` | 17/17 | gather/scatter, masking, atomics |
> | `test_reductions` | 14/14 | block reduce / prefix reduce / compress |
> | `test_vcall` | 14/14 | dynamic dispatch, recursion, side effects |
> | `test_record` | 9/9 | frozen-function recording and replay |
> | `test_array` | 15/15 | per-lane variable arrays |
>
> Registering them was worth more than any bespoke test: it found nine emitter
> bugs and six wiring bugs, most of which compiled and ran while computing the
> wrong thing (BACKEND_NOTES §11f). Two classes are worth carrying forward — a
> `JitBackend` packed into a 2-bit field truncates HIP silently (now guarded by
> `tests/hip_packing.cpp`), and under the shim every allocate/launch/**sync**
> site must treat HIP as CUDA-backed.
>
> **Done — pure components, each test-first with a contract test:**
> | Component | Source | Test |
> |---|---|---|
> | Type mapping | `src/hip_eval.h` | `tests/hip_types.cpp` |
> | Literal materialisation | `src/hip_literal.h` | `tests/hip_literal.cpp` |
> | Kernel prologue | `src/hip_prologue.h` | `tests/hip_prologue.cpp` |
> | Source reindent | `src/hip_format.h` | `tests/hip_format.cpp` |
>
> **Done — opcode specification.** `tools/hip_validate/kernels/spec_*.hip` hand-write
> the source the emitter must produce for ~45 VarKinds (arithmetic, compare/select,
> bitwise, casts, transcendentals, memory, wave ops, wide multiply), shaped as machine
> output. Self-checking via `--expect-zero`, and every one is proven to compile for
> real `gfx90a` **and** compute correctly. 16 passed / 0 failed.
>
> **Remaining in Phase 2:** half-precision atomics (no 16-bit `atomicCAS`; needs
> the packed `f16x2` treatment the CUDA backend uses) and the ray-tracing opcodes,
> which belong with Phase 0b. Warp pre-aggregation for scatter-reduce and
> `ScatterInc` is deferred as a contention optimisation, not a correctness gap.
>
> **Every Dr.Jit-Core test suite now passes against HIP.**
>
> **Constraints discovered, binding on the emitter** — full detail in BACKEND_NOTES
> §11a/§11c, summarised here because each one compiles cleanly and is still wrong:
> - Emit CUDA/HIP device intrinsics, **never Clang `__builtin_*`** — hipcc accepts them,
>   NVRTC does not, and they pin generated source to one compiler.
> - `Round` is `rintf` (nearest-**even**), not `roundf` (half away from zero).
> - `__exp2f` does not exist on either platform; use `exp2f`.
> - **No `__syncthreads()` below the prologue's bounds-check `return`** — undefined,
>   and fails at *launch*, not compile.
> - **Atomics must target global memory**, never a materialised temporary — likewise a
>   launch-time failure.
> - Approximation ops are not bit-exact across backends and must be pinned with
>   tolerances, with accurate and approximate forms held to *different* ones.

`hip_eval.cpp`: emit HIP C++ source per §3.1's defensive-emission rule, compile via
`hiprtc`, launch via `hipModuleLaunchKernel`. Bring opcodes up incrementally: arithmetic →
compare/select → memory (gather/scatter) → control flow (loops, `if`) → dynamic dispatch
(calls).

**Codegen tuning is an explicit work item, not a default.** Dr.Jit already optimizes at the
graph level, then `hiprtc` re-runs its own pipeline over an already-optimized megakernel
with thousands of live values — which can inflate register pressure and cost occupancy.
(Note this is a difference of *degree*, not kind: `ptxas` also re-optimizes the CUDA
backend's PTX. Clang's frontend is simply slower and more aggressive.) Concretely:
- **Measure occupancy across `-O1` / `-O2` / `-O3`** on real Mitsuba kernels rather than
  assuming `-O3`.
- **Use `__launch_bounds__`** to bound register allocation — the first-class HIP mechanism
  for exactly this problem.
- **Track cold-compile time per megakernel.** This is the one metric that could justify
  revisiting §3.1; the `~/.drjit` cache hides it from users but not from us during this
  phase, where every codegen change invalidates every cached kernel.

*Milestone: `c = (a + b) * 5` correct on MI210, then the loop/call tests pass.*

### Phase 3 — Device library ✅ *(written; wave64 semantics unverified)*
`hipcc` rule for `resources/kernels.cu` → `gfx90a` code object; wave64 audit of the
`.cuh` sources per §3.3; embed/decompress path. Wire `block_reduce`,
`block_prefix_reduce`, `reduce_dot`, `compress`, `block_mkperm`.

**Done.** `resources/*.cuh` was ported **in place** rather than forked, so the CUDA and
AMD targets cannot drift. `WarpSize` is now §3.3's single-definition parameter on both
sides, and the two idioms that hid a width behind a literal — `31 - __clz(ballot)` and
`peers << (32 - lane)` — have named, width-correct replacements. ROCm `static_assert`s
that ballot masks are 64-bit, so a site missed here is a compile error rather than a
wrong answer. `resources/Makefile` builds the code object (hipcc `--genco` →
`clang-offload-bundler --unbundle` → `pack_hip`, 1.38 MB → 168 KB); the blob is
committed, so building drjit-core still needs no ROCm. The six `HIPThreadState` methods
launch it, with every warp-size constant read from `HIPDevice` — four sites in the launch
configurations where the CUDA original's literal 32 would miscompute on gfx90a.

**Two independent checks that the port did not break anything**, both at width 32:
entry-by-entry nvcc PTX comparison (638 of 646 byte-identical; the 8 that differ are the
intended ones — 7 mkperm plus `compress_large`), and
`tools/hip_validate/devlib_check.sh`, which rebuilds the CUDA blob from the ported
sources and runs drjit-core's own suite against it (8/8). The normal suite cannot see the
port at all, since drjit-core embeds the *committed* blob, which predates it.

**Wave64 remains unverified** — that is the whole reason the width is a parameter and not
a constant. The host also refuses to load the blob when the device's reported width
differs from the one it was compiled for, so a mismatch fails loudly instead of returning
quietly wrong reductions.
*Milestone (outstanding): reductions, scans, and compress numerically correct under
wave64.*

### Phase 4 — Advanced ops ✅ *(bar fp16 atomics)*
Scatter/atomics (wave64 ballot redesign), packet memory, textures.

**Two of the three turned out not to be work at all.**

**Textures need nothing.** [texture.h:43](ext/drjit/include/drjit/texture.h#L43) gates
every hardware path on `HasGPUTexture = (IsHalf || IsSingle || IsUInt8) && (IsCUDA ||
IsMetal)`. A HIP array is neither, so every `if constexpr (HasGPUTexture)` branch
compiles out and `dr::Texture` takes its software path automatically. `TexLookup`,
`TexFetchBilerp` and `TexWrite` are never emitted for this backend, and Phases 6–7 are
not blocked on them. **When Layer B adds `is_hip_v`, do not add `IsHIP` to that
disjunction** — it would switch on a `jit_hip_tex_*` path that does not exist. Note also
that CUDA's texture units resolve sub-texel position with only 8 fractional bits, so the
software path is *more* accurate, merely slower.

**Scatter aggregation is not a correctness gap.** §3.3 flags `cuda_scatter.cpp`'s peer
aggregation as needing a 64-wide ballot redesign. It does — but only if we want it. The
HIP backend issues plain per-lane atomics, which are always correct and merely slower
under contention. There is no wave64 ballot to redesign because there is no ballot, so
this is a performance task for after the backend is correct.

**Packet memory** is implemented, and needs none of the wide-vector machinery Metal and
CUDA carry: LLVM merges the element-wise form into the same `global_load_dwordx4` an
explicit `float4` produces, measured on gfx90a. `BoundsCheck` (debug mode) is implemented
too — it was the one gap that would have made the mode you bring a backend up in the one
mode that could not run.

*Remaining:* fp16 atomics (packed `f16x2`). `render_scatter_reduce` fails on Float16 —
there is no 16-bit `atomicCAS`, so there is no correct lowering — and
`jitc_can_scatter_reduce()` now **reports that**, where before it inherited a generic
"yes" and aborted halfway through codegen. No variant we target reaches it.

*Also found, and deliberately left:* `op.cpp`'s `use_packet_op` selection has arms for
LLVM, CUDA and Metal and none for HIP, so a packet scatter-**reduce** decomposes into
scalar ones. Correct, and costs nothing today: what makes the packet form worth having on
CUDA is `ReduceMode::Local`, which this backend does not implement. The emitter handles
the case anyway, so adding the arm is a one-line change when Local lands. This is the
second per-backend if/else chain in shared code found to omit HIP silently (§9).

### Phase 5 — Ray tracing ✅ *(emitted, gfx90a-linked, and executed)*
Implement `jit_hip_ray_trace` + `HIPScene` per §3.2, mirroring the Metal pair. Host-side
BVH build via the HIP-RT API; scene handle bound as a kernel parameter. Replace the
`jitc_cuda_render_trace` role
([cuda_eval.cpp:1244](ext/drjit/ext/drjit-core/src/cuda_eval.cpp#L1244)) with an inline
HIP-RT traversal — no raygen entry, no SBT, no callables. Custom intersection functions
for Mitsuba's implicit shapes go through `hiprtFuncTable`, mirroring
`MetalScene::intersection_fns`.

**Done, and the signature is Metal's verbatim after all.** §7a warned that
`hiprtHit` carries six of Metal's eight outputs, so "adopt the signature verbatim"
might need qualifying. It did not: `geometry_id` and `user_instance_id` come from device
tables indexed by the hit's instance ID, which is *exact* rather than a workaround —
a HIP-RT instance references exactly one geometry, so both really are properties of the
instance. Metal needs per-hit fields because its acceleration structures nest geometries
inside an instance. So `jit_hip_ray_trace` is byte-for-byte `jit_metal_ray_trace`, which
is what makes Layer C cheap.

`HIPScene` is much smaller than `MetalScene`: a `hiprtScene` is a plain device pointer
travelling in the parameter block, so there is no resource-handle reconstruction, no
per-launch residency list and no retained intersection-function library.

**And it executes, here.** `jitc_hip_compile()` routes any kernel containing the HIP-RT
include through `hiprtBuildTraceKernels()` rather than bare NVRTC — that call compiles
*and links* the traversal library, and on NVIDIA it drives NVRTC underneath.
`tests/hip_trace_exec.cpp` builds a real BVH and scene, traces through
`jit_hip_ray_trace()`, and checks the results: 32 lanes inside a triangle hit at
t = 1.0, 32 outside miss, and `geometry_id` arrives from the instance-indexed table.
Traversal correctness is no longer owed to the MI210 — gfx90a has no RT hardware, so it
takes the same RTIP 0 software path NVIDIA just ran (§7b).

Codegen is also checked two other ways: `tests/hip_trace.cpp` captures the *real* emitted
source through the `PrintIR` log callback and asserts its structure, and `run_tests.sh`
compiles that for real gfx90a — 64 VGPR / 38 AGPR / 784 B scratch, within noise of §7a's
hand-written traversal.

**§7a finding 3 was corrected in the process,** and it matters for anyone reading it:
who defines `intersectFunc`/`filterFunc` depends on how HIP-RT is linked. Hand-linking
the bitcode (what `hip_validate` does) requires the application to supply them;
`hiprtBuildTraceKernels()` — what the backend uses, shim and hardware alike — generates
them and rejects a duplicate. Emitting stubs, as §7a concluded, breaks the backend. The
general lesson: §7a's observations were made through the harness's link path, which is
not the backend's.

*Remaining:* custom-primitive dispatch through `hiprtFuncTable`. A scene built with one
raises at codegen rather than silently reporting every custom shape as a miss; wiring it
up means passing the geometry and ray types through to `hiprtBuildTraceKernels()`.
*Milestone (outstanding): `drjit-core` RT tests pass on MI210 — i.e. at wave64.*

### Phase 6 — Dr.Jit layer (Layer B)
`drjit.hip` / `drjit.hip.ad` namespaces; `is_hip_v` traits. Verify `@dr.freeze` works —
`record_ts.cpp` has backend-predicate coupling, and Mitsuba has a 16-hit
`test_freeze.py` plus freeze-specific handling in `scene_optix.inl` and the
`DRJIT_TRAVERSE` handle pattern in
[accel_native.h](include/mitsuba/render/accel_native.h). Small, but it will otherwise
surface as mystery failures later.

**"Small, but it will otherwise surface as mystery failures later" was right, and
understated.** The namespaces, `IsHIP`/`is_hip_v` traits and `HIPArray`/`HIPDiffArray`
aliases are mechanical. What was not: **every** bug in this phase was silent.

`drjit.hip` came out empty, and that one symptom had *two* independent causes, either
sufficient on its own — `detail::backend<T>` had no HIP specialisation (so
`backend_v<HIPArray<float>>` was `None` and every type was filed under `drjit.scalar`),
and `ArrayMeta::backend` was a 2-bit field (so `HIP == 4` truncated to 0). Fixing either
alone leaves the symptom unchanged.

That is the **third** packed backend field found one bit too narrow, after
`Variable::backend` and `AllocInfo`; it now carries a `static_assert` against
`JitBackend::Count`. Widening it pushed `ArrayMeta` from 8 to 12 bytes — neither
neighbour could spare a bit — which in turn broke `operator==`'s whole-struct `memcmp`
and `meta_get_type()`'s `uint64_t` cache key. Both now share a `meta_identity()` helper,
which is *more* correct than what it replaced: the new second word is nearly all padding,
and padding is indeterminate.

**Thirteen per-backend chains in shipped code omitted HIP** in this phase alone, plus six
more in the test suite. The instructive one is `dlpack.cpp`: HIP fell through to *host*,
so a device pointer would have been handed out labelled CPU — a segfault in the consumer's
process, not ours. Those now go through `is_device_backend()` in `src/python/common.h`.

Two of them are worth naming because they fail in ways that mislead. `Resampler` was
missing HIP at *four* layers at once, one of which was a `-DDRJIT_ENABLE_HIP` that
`src/extra/CMakeLists.txt` re-derives per target — so the instantiations were correct,
looked correct, and were compiled out, surfacing as an `undefined symbol` for a template
you can watch being instantiated. And `jitc_coop_vec_supported()` defaulted to **yes**,
so an unimplemented feature reached `jitc_fail()`, which *aborts* — killing the entire
pytest process rather than reporting one skip. A capability table whose fallthrough is
"supported" is a trap for every backend added after it was written.

**The most expensive bug was not a codegen bug at all.** A `Warn`-level log in the shim's
`jit_hip_init()` deadlocked the interpreter at exit: `jit_init_async` runs backend init on
a background thread holding `state.lock`, Warn reaches the Python log callback, which
takes the GIL, and the main thread holds the GIL waiting on `state.lock`. Everything
worked and printed correct results; the process just never terminated. Backend init must
log at `Info`. See BACKEND_NOTES §11k.

`@dr.freeze` needed no work. A test asserting "traces once, then replays" reported four
traces on HIP — and four on CUDA and LLVM. The expectation was wrong, not the backend.
**Compare against a reference backend rather than an absolute**, or the next person hunts
a HIP bug that does not exist.

**For Phase 7, grep before building.** `JitBackend::CUDA`, `is_cuda_v`, `MI_ENABLE_CUDA`,
`backend == `. The sites that mean "this is a GPU" rather than "this is CUDA
specifically" are the ones that will be wrong, and none of them will say so.

**Status: met.** Dr.Jit's suite is green on `hip` / `hip.ad` — ~19,000 passing across all
44 test files, parameterised over HIP, HIP.ad, CUDA, CUDA.ad, LLVM, LLVM.ad and scalar,
with CUDA and LLVM unregressed by the `ArrayMeta` widening.

**The last crash was ours, and the investigation nearly recorded it as upstream's.**
`test_freeze.py::test72_no_input[llvm]` segfaulted in `__dynamic_cast`. It reproduced with
HIP switched off, which looked like proof it was not us — but "HIP off" is not "our
changes off", and the Phase 1–5 drjit-core commits were still in that build. Building the
actual upstream merge-base (`9a7db92b`, submodule `7a9ab1fa`) was the test that settled
it: 744 passed, no crash. **A control has to differ in exactly the variable you are
testing.** See BACKEND_NOTES §11l for the bug itself — a refactor of mine that flattened
a per-backend if/else chain and, in doing so, silently reversed a load-bearing ordering.

Run the suite per-file, not as one `pytest tests/`: a native crash aborts the whole run,
so a segfault at 22% tells you nothing about the other 78% — and it hid two *further*
crashes behind it here. `scratchpad/suite_perfile.sh` reports `CRASH rc=139` per file and
keeps going.

**And a skipped test is not a passing one.** The suite read green while 497 tests never
ran: the seven C++ extension suites need `DRJIT_ENABLE_TESTS=ON` (off by default), a
`DRJIT_ENABLE_HIP` define that `tests/CMakeLists.txt` did not pass, and a `get_pkg()`
that did not stop at Metal — three independent reasons, each sufficient, and all of them
printing as `s`. (Four reasons, in the end — an eighth `get_pkg()` copy lived in
`test_freeze.py`, which is not an `_ext` file and so was not on the list.)

They now run and pass on HIP, and that is the most useful result in this phase for
Phase 7: `call_ext` is the C++ side of vcall dispatch, and unlocking it also unlocked
~120 tests in `test_freeze.py`, among them the whole frozen-vcall set. **HIP vcall
dispatch demonstrably works end to end** — §9a-c's risk area, previously untested.

Run the sweep from `build-hip/tests`, where the extension `.so`s live, and re-run cmake
after editing `tests/*.py` (they are copied at configure time). See BACKEND_NOTES §11m.

*Milestone: Dr.Jit's own test suite green on `hip` / `hip.ad`.*

### Phase 7 — Mitsuba layer (Layer C)
`accel_hip.h` + `scene_hip.inl` driven by `SceneIRBuilder`; `SceneAccel` trait branch;
`MI_ENABLE_HIP` + variant registration; generalize the ~20 `is_cuda_v` /
`MI_ENABLE_CUDA` guard sites; compile out `optixdenoiser` **and exclude its tests**
(§5.9 gap 2). **Wire the four test-harness sites in §5.9** — after which the whole suite
runs on HIP automatically.

> **⏳ IN PROGRESS — traversal, attribution and three custom shapes done; the suite is not yet green.**
>
> **Milestone met, with a correction to what it proved.** `hip_ad_rgb` renders a mesh
> scene matching `llvm_ad_rgb` to **max |diff| 1.19e-07**, with HIP-RT's own log
> confirming the real path: `createGeometry → buildGeometry → createScene → buildScene`.
> LLVM traces through Embree and certainly sees the geometry, so HIP-RT agreeing to 1e-7
> means it is intersecting rather than missing. **That inference holds.**
>
> What it did **not** prove is per-shape attribution. The scene had one mesh, and
> `SceneIR` buckets same-kind geometry into a single BLAS — so every shape-table base was
> `0`, and an out-of-bounds read of the recovery table returned the correct answer by
> coincidence. A two-mesh scene had been mis-shading every hit after the first since this
> phase began. Fixed; see BACKEND_NOTES §11n.2, and `test15_many_top_level_meshes`, which
> covers the contract with meshes only so no backend can skip it.
>
> The reusable form: **a scene with one of something cannot distinguish an index from a
> base.**
>
> **Done.** `MI_ENABLE_HIP` + the 24 `hip_*` variants; `mitsuba::is_gpu_v`; the
> `SceneAccel` branch, `accel_hip.h`, `scene_hip.inl` and `hip/accel.{h,cpp}`;
> `jit_hip_rt_context()` in drjit-core; the four §5.9 harness sites.
>
> **§5.9 gap 2 needed no work.** The denoiser is already triple-guarded — CUDA-only
> pytest fixtures, `MI_ENABLE_CUDA` in CMake, and a constructor `Throw`.
>
> **Six wiring bugs, every one of which fails late or silently:** Mitsuba never forwarded
> `DRJIT_ENABLE_HIP`; it read drjit-core's `DRJIT_HIPRT_PATH`, which is only set inside
> the *shim* branch and so would have failed on the MI210 itself; `PRIVATE` link options
> on an OBJECT library never reach the consuming link; `drjit_v.cpp` would have imported
> `drjit.scalar` for a HIP variant; and the color-space tables had neither a HIP arm nor a
> HIP flag in their initializer, so `get_color_space_tables<HIP>()` returned the **scalar**
> tables. That last one happened to fail at compile time. That was luck — the same
> omission in a non-template context is a host pointer handed to a device kernel, and it
> is now the worked example in `is_gpu_v`'s documentation of where *not* to use the trait.
>
> **The §5 count of "~20 `is_cuda_v` sites to generalize" was misleading** — 31 exist, and
> they are four different questions. 15 genuinely ask "CPU path?" or "host-addressable
> memory?" (→ `is_gpu_v`); 4 are genuinely OptiX-specific and already correctly inside
> `MI_ENABLE_CUDA` (→ unchanged); 2 select a per-backend *resource* and need real arms,
> not a trait; and `sphere.cpp`/`cylinder.cpp` test `is_cuda_v<FloatP>` on a **packet**
> type, which is always false and unreachable on GPU variants — dead, deliberately left
> alone rather than perturbing numerics for no gain.
>
> **Suite baseline (144 files, HIP variant only), re-measured after custom primitives:
> 757 passed, 74 files fully green, 48 with no HIP-parameterized tests, 14 with
> problems** (was 743 / 69 / 48 / 16). `test_renders.py` is **190 passed / 6 failed**,
> and all 6 are one scene refused for its `sdfgrid`.
>
> Every one of the 14 is attributed:
>
> | Cause | Files | Status |
> |---|---|---|
> | `hiprtBuildTraceKernels` crash | test_ad, test_aov, test_ad_integrators, test_freeze, test_mesh, test_instance, test_ptracer | **one bug, seven symptoms** — and it is NONDETERMINISTIC (§11n.1a): same binary, same command, SEGV / abort / SEGV. Do not classify these by stack signature. |
> | sdfgrid not implemented | test_sdfgrid, test_renders (all 6) | refused by type, message generated from the capability table |
> | ellipsoids not implemented | test_volprim_rf_basic, test_ellipsoidsmesh | refused by type |
> | curves not implemented | test_bsplinecurve, test_linearcurve | refused by type |
> | Marginal statistics | test_hair `test06_chi2` | p=0.009929 vs α=0.01, 22/23 checks accepted — a false positive is ~21% likely across 23 tests at that threshold. Flagged, not "fixed". |
>
> **A refusal is not a free pass.** `test14_many_top_level_analytic_shapes` was in the
> "refused by design" column of the previous baseline. It was also the only test checking
> per-shape attribution, and it built its scene from spheres *and* disks — so the §11n.2
> bug sat behind what read as a known limitation. When a backend refuses a feature, check
> what else the refusing tests were measuring (§11o.3).
>
> **Two things remain, and they are independent.**
>
> **1. Custom/implicit geometry — DONE for sphere, disk and cylinder.** AABB-list
> geometry, a `hiprtFuncTable`, per-primitive data upload, and device intersectors in
> `src/render/hip/intersection_functions.hip` (text embedded into libmitsuba-render and
> registered through the new `jit_hip_set_isect_source()`; HIP-RT generates its dispatcher
> inside `hiprtBuildTraceKernels()`, so what must reach the runtime is *source*, not an
> object). The signature HIP-RT expects is in none of its headers — settled by
> `tools/hip_validate/isect_probe`, which recovers it from the shipped library and proves
> it by building both a correct and a deliberately wrong version (BACKEND_NOTES §11o.1).
>
> Data location is simpler than Metal's: HIP-RT passes one pointer per geometry *type*
> and `hit.instanceID` is live on entry, so one instance-indexed base table replaces
> Metal's two-level lookup — exact, because a HIP-RT instance references exactly one
> geometry.
>
> **Still refused, by type, naming themselves: ellipsoids, sdfgrid, and curves.** These
> are the ones that are not transcription — variable-length per-shape data, where Metal
> fills per-ellipsoid records and OptiX hands over two device pointers plus a precomputed
> AABB buffer. That is a design choice, not a port.
>
> **2. One bug, seven symptoms — and it is random.** Every remaining crash has the same
> root cause: `hiprtBuildTraceKernels()` fails on kernels that contain a trace **and**
> come from a symbolic loop (`ad_loop`) or a vcall (`jit_var_call_reduce`) — exactly the
> kernels a real integrator emits.
>
> The emitted source is **exonerated**: the exact text drjit-core logs on failure builds
> successfully standalone through `hiprtBuildTraceKernels` with argument-for-argument
> identical parameters (`tools/hip_validate/rtrepro`). Also ruled out by experiment:
> cumulative resource exhaustion (12 build/destroy cycles clean, 8 distinct traced kernels
> clean), missing CUDA context binding (a real bug, fixed, crash survives), and device
> memory (5.6 GB of 6 GB free at the crash).
>
> **New, and the most useful fact about it: the failure is nondeterministic.** Same file,
> same command, same binary — SEGV / abort / SEGV, always at the same test, with
> `libhiprt` frames present in some runs and absent in others. So it is not the arguments
> either; what is left is state — a race, a use-after-free, or an uninitialized read,
> ours or HIP-RT's. Two consequences: never classify these crashes by stack signature
> (that is a coin flip), and **any experiment against this bug needs repetition, because
> a passing run proves nothing**. **Not assumed to be shim-only** —
> `hiprtBuildTraceKernels()` is the API the backend uses on real hardware too.
> See BACKEND_NOTES §11n.1 and §11n.1a.

*Milestone: `hip_ad_rgb` renders a scene matching `llvm_ad_rgb` within tolerance.*

### Phase 8 — Spectral bring-up & validation
`hip_ad_spectral`. Spectral is a Mitsuba-level variant — more float lanes through the
same kernels, no new opcode — so if Phase 7 is solid this is mostly a build + validate
step, dependent on the texture path from Phase 4.
Run the full suite against HIP (§5.9: 136 Mitsuba test files + 52 Dr.Jit test files, plus
`test_renders.py`'s z-test over 13 scenes × rgb/spectral). Because the render tests compare
statistically against color-mode-keyed references rather than per-backend images, no
reference regeneration is required. Cross-check numerically vs. LLVM and CUDA. Then
profile: `rocprof` on the traversal path to quantify the §7.3 software-BVH cost for the
fishsense workload, using the workload-representative assets developed separately (§5.9).
*Milestone: fishsense pipeline runs end-to-end on MI210.*

---

### 5.9 Test infrastructure — inventory, wiring, and gaps

**Verified inventory in this checkout:**

| Suite | Count |
|---|---|
| Mitsuba test files | 136 (97 test functions, heavily variant-parameterized) |
| Dr.Jit Python tests | 52 files |
| drjit-core C++ tests | ~19 (`basics`, `loop`, `reductions`, `vcall`, `record`, `mem`, `array`, …) |
| Reference render scenes | 13, with **41 rgb + 41 spectral** reference EXRs |

**The suite auto-enrolls a new backend.** [conftest.py](src/conftest.py) generates variant
fixtures dynamically from `mi.variants()`. Groups matched purely on suffix — `all_rgb`,
`vec_rgb`, `vec_spectral`, `all_ad_rgb`, `all_ad_spectral` — pick up `hip_ad_rgb` and
`hip_ad_spectral` with **zero changes**.

**Render tests are already designed for cross-backend validation.**
[test_renders.py](src/render/tests/test_renders.py) does not diff images. It runs a
**z-test against reference mean + variance EXRs, keyed by color mode rather than backend**
([:159-165](src/render/tests/test_renders.py#L159-L165); refs resolved at
[:107-114](src/render/tests/test_renders.py#L107-L114)). That is a statistical test that the
new backend draws from the correct distribution — the right tool here, and it means **no
reference regeneration is needed**. Spectral references are complete, 1:1 with rgb across
every scene.

**Wiring required — four sites, ~15–20 lines total (do this in Phase 7):**
1. [conftest.py:82-89](src/conftest.py#L82-L89) — the `available` filter needs a `hip` arm
   plus `dr.has_backend(dr.JitBackend.HIP)`.
2. [conftest.py:99](src/conftest.py#L99) — `all_possible_variants` iterates
   `["llvm", "cuda", "metal"]`.
3. [conftest.py:106-124](src/conftest.py#L106-L124) — add `any_hip`; extend
   `all_backends_once`, `vec_backends_once`, `vec_backends_once_rgb`,
   `vec_backends_once_spectral`.
4. [test_renders.py:71](src/render/tests/test_renders.py#L71) and
   [:184](src/render/tests/test_renders.py#L184) — the `is_jit` backend checks.

**Four gaps:**

1. **No `hip_triangle.cpp`.** drjit-core ships `optix_triangle.cpp` and
   `metal_triangle.mm` as RT smoke tests. Write the HIP equivalent — and write it **as** the
   Phase 0b spike rather than after it, since it is exactly the shape that spike must prove
   (generated kernel + HIP-RT traversal + one custom intersection callback).
2. **Exclude the denoiser tests** — `test_optixdenoiser.py` and
   `resources/data/tests/denoiser` — since §2 drops the denoiser for HIP variants.
3. **`volprim_rf_basic` has no reference images** (0 rgb, 0 spectral). Determine whether it
   is skipped or generates references on the fly: volumetric primitives exercise the
   custom-shape path that HIP-RT's `hiprtFuncTable` must support, so it should not be
   silently absent from coverage.

**Assessment:** coverage is sufficient and the infrastructure was clearly built for
multiple backends (the Metal work will have proven this out). The 13 bundled scenes cover
generic rendering — `bsdf_spheres`, `instancing`, `participating_media`, `various_shapes` —
which validates correctness but not the production workload. Workload-representative test
assets are being developed separately and are **out of scope for this plan**; Phase 8's
end-to-end validation assumes they exist by then.

---

## 6. Estimate

Roughly **4–7 months for one engineer**, with ray tracing dominating the variance.

| Phase | Estimate |
|---|---|
| 0 — wiring + RT spike | 2–3 wks |
| 1 — runtime/driver | 2–3 wks |
| 2 — codegen | 6–10 wks |
| 3 — device library | 2–3 wks |
| 4 — advanced ops (software textures) | 3–4 wks |
| 5 — ray tracing | 4–8 wks |
| 6 — Dr.Jit layer | 1–2 wks |
| 7 — Mitsuba layer | 2–4 wks |
| 8 — spectral + validation + perf | 3–5 wks |

**Sequencing principle: get compute-only working before touching RT.** Phases 1–4 and 6
deliver a fully functional non-RT `hip` backend, which exercises all the autodiff
machinery and a large fraction of Mitsuba. That is a solid correctness baseline to stand
on before the risky phase — and it is independently useful.

---

## 7. Risks & gates

### 7.1 Codegen + `hiprtc` integration (highest effort, moderate risk)
`cuda_eval.cpp`'s ~1,900 lines of per-opcode templates are a genuine rewrite. Mitigation:
`metal_eval.cpp` (1,675 L) is a recent from-scratch source-emitting backend — read it
first, and keep the emit layer abstract per §3.1. Codegen errors are largely catchable
**without a GPU**: emit source, compile with `hipcc --genco --offload-arch=gfx90a`, and
verify it builds and the intrinsics are valid.

### 7.2 Wave64 (moderate risk, wide blast radius)
Ballots and shuffles are not mechanically substitutable (§3.3). Mitigation: audit
before Phase 3 rather than debugging numerically wrong reductions afterward. `gfx90a`
being wave64-only removes an entire class of variability.

### 7.3 HIP-RT on `gfx90a` — ~~highest risk~~ **RESOLVED (§0.2)**
**HIP-RT 3.0.3 supports `gfx90a`**, confirmed locally: `rocmPackages.hiprt` is packaged
and unbroken, and `hiprt_common.h` handles `__gfx90a__` explicitly. The availability risk
is closed and this is no longer the project's highest risk — that title passes to §7.1
(codegen effort) and §7.2 (wave64).

**The performance caveat stands, unchanged.** HIP-RT assigns `gfx90a` `HIPRT_RTIP 0` (the
`#else` branch; tiers are 3.1/2.0/1.1 for RDNA2+), i.e. **software BVH traversal, no
hardware RT**. Whether that is fast enough for the fishsense workload is a Phase 8
measurement, not an availability question.

The fallbacks below are retained as contingency should traversal prove too slow in
practice, but they are no longer on the expected path.

**Fallbacks, in preference order, if HIP-RT does not clear the gate:**
1. **Port Mitsuba's own kd-tree.** [scene_native.inl](src/render/scene_native.inl) +
   [kdtree.cpp](src/render/kdtree.cpp) is already backend-neutral CPU code Mitsuba uses
   when Embree is off. Fastest route to a correctness baseline; a kd-tree is a poor
   SIMT fit, so expect weak performance.
2. **Write BVH traversal in Dr.Jit IR.** Fully portable across backends, integrates
   naturally with the existing custom-shape intersection code, no HIP-RT dependency.
   We own the builder and the traversal loop.

Separately: **software traversal on MI210 may simply be too slow** for the fishsense
workload even if HIP-RT works. Measure in Phase 8 — but be aware this is a property of
the card, not of our implementation, and no amount of engineering recovers hardware
that isn't there.

### 7.4 `hiprtc` compile latency (low risk — now quantified, §0.2)
A full Clang front-end per kernel. Mitigation: the existing `~/.drjit` disk kernel cache
with a HIP tag (§8.4) absorbs it in steady state. If it proves fatal, §3.1's emit
abstraction keeps the IR + comgr route open.

### 7.5 Upstream drift (moderate, manageable)
The Metal backend landed *very* recently — `drjit-core` HEAD includes a Metal codegen
fix (`26d4a14`), and the multi-backend abstractions are still in flux. A long-lived fork
will hurt. See §9.

---

## 8. Open questions — resolve before Phase 1

0. **What shape is fishsense's parallelism — many independent tasks, or one render?**
   **BLOCKING, and unanswered.** If it is thousands of independent problems, the phone
   fleet on `llvm_ad_spectral` (zero engineering) may suffice and *both* GPU tracks drop to
   nice-to-have. This question sizes the entire project and should be answered before
   week 1, not at the month-6 review. See §0.
1. ~~**Does HIP-RT support `gfx90a`?**~~ **ANSWERED: yes** — HIP-RT 3.0.3, verified
   locally (§0.2). `gfx90a` gets `HIPRT_RTIP 0`, so traversal is software-only; that is a
   Phase 8 performance question, not an availability one.
2. Confirm the ROCm version, `hiprtc`, and comgr availability on the MI210 host once it
   lands. An MI210 is being procured; until then development runs on the local NVIDIA box
   per §0.3, which does not block Phases 0a, 2 or 3.
3. FP64: MI210 is full-rate double precision, so the double path is first-class rather
   than an afterthought. Which Mitsuba spectral kernels should be numerically validated
   against CUDA/LLVM first?
4. Disk kernel cache: reuse `~/.drjit` with a HIP/`gfx90a` tag in the cache key.
5. Does the fishsense workload's scene complexity make software BVH traversal viable at
   all? Get a rough bound early — a back-of-envelope traversal-cost estimate before
   Phase 5, not after.
6. **Does HIP-RT work under HIP-on-CUDA?** If yes, part of Phase 5 develops on the local
   NVIDIA card too (§3.5). Nice-to-have, not load-bearing — do not plan around it until
   confirmed.

---

## 9. Code organization & repo hygiene

drjit-core has **no runtime plugin architecture**. Backends are compile-time-woven: the
`JitBackend` enum is fixed, codegen dispatch is a hardcoded `#if defined` chain
([eval.cpp:589-599](ext/drjit/ext/drjit-core/src/eval.cpp#L589-L599)), and ThreadState
construction is a switch in `init.cpp`. The `dlopen` calls in the tree load vendor
*runtimes*, not backend plugins. A zero-diff pluggable backend is therefore not possible
without first adding a registry seam to the core.

**Approach: thin upstream seam + out-of-tree backend files.**
- **Bulk of the backend lives in new `hip_*.cpp/.h` files we own** — new files never
  conflict on `git merge upstream/master`.
- **Irreducible upstream patch, all guarded by `DRJIT_ENABLE_HIP`** (~100 lines): enum
  entry + `jitc_is_hip` in `jit.h`; ThreadState construction and init/shutdown/sync in
  `init.cpp` (~5 sites); `jitc_hip_assemble` + compile dispatch in `eval.cpp` (~2 sites).
- **Dispatch inheritance:** the 18 `jitc_is_gpu` sites come free once HIP is defined as
  a GPU backend. Of the 54 `jitc_is_cuda` sites, many are really "is this the
  codegen/GPU path" and fold into shared predicates; only true CUDA-vs-Metal branches
  need a HIP arm. The 46 `jitc_is_metal` sites are untouched.
- **Optional seam-shrink, plausibly upstreamable:** add virtual `assemble()` /
  `compile()` to `ThreadState` so `eval.cpp` dispatches generically and the codegen
  branch leaves the patch entirely. Maintainers have visible reason to accept this —
  they just added Metal.

**Branch setup.** Three nested repos need this treatment: `mitsuba3`,
`ext/drjit`, `ext/drjit/ext/drjit-core`. For each: add our own `origin`, keep
`mitsuba-renderer` as `upstream`, do HIP work on a branch, and merge `upstream/master`
periodically. Given §7.5, merge on a schedule rather than when it becomes painful.

**Consider engaging upstream early.** A HIP backend is exactly the kind of contribution
that may be welcomed, and coordinating with the Metal work in flight is far cheaper than
rebasing against it for six months.

---

## 10. Next actions

**✅ Completed on the local NVIDIA box — no AMD hardware involved:**
1. ~~Answer §8.1~~ — HIP-RT supports `gfx90a` (§0.2).
2. ~~Read `metal_eval.cpp` / `metal.h` / `scene_metal.inl` end to end~~ — distilled into
   `tools/hip_validate/BACKEND_NOTES.md`, the `hip_eval.cpp` blueprint.
3. ~~Phase 0a wiring~~ — backend registered but inert, locked by `tests/hip_wiring.cpp`.
4. ~~Stand up the dual-validation harness~~ — `tools/hip_validate`, both arms working.
5. ~~Phase 2 pure components~~ — types, literals, prologue, reindent; all test-first.
6. ~~Opcode specification~~ — `spec_*.hip` for ~45 VarKinds, self-checking, 16/16 green.
7. ~~Wave64-unverified list~~ — now emitted by `run_tests.sh` on every run rather than
   kept in prose.
8. ~~Phase 2 emitter~~ — `jitc_hip_render()` and the variable loop, verified by the
   upstream suites rather than a bespoke test. `tests/hip_codegen.cpp`'s `WILL_FAIL`
   tripwire fired and was removed; it is now a regression guard.
9. ~~Control flow~~ — `test_loop` passes 9/9. It had been written-but-unexercised.

**⏳ Remaining — most of it now doable locally after all (§3.5):**
0. **Phase 1 is UNBLOCKED.** HIP-on-CUDA runs the real HIP API on the 3060, so
   `hip_api.cpp` / `hip_ts.cpp` can be written and tested here. Only AMD-specific
   behaviour still needs the card.
1. **On MI210 arrival, re-run the unverified list first.** `run_tests.sh` names them: currently
   `smoke_half`, `smoke_literal`, `smoke_types`, `smoke_warp`, `spec_wave`. These are
   cheap, and they are where a latent wave64 or fp16 bug will surface. **Then the device
   library** (`test_reductions`, `test_mem`, `test_vcall`): its wave-width rewrite is
   verified only at 32, and its failure mode is a wrong number rather than a crash.
2. ~~Phase 1~~ — `hip_api` (dlopen bindings, ABI-checked against real ROCm),
   `hip_core` (device enumeration) and `hip_ts` (HIPThreadState) are written. Symbol
   names and constants are verified against the installed ROCm; every ThreadState
   call SHAPE is verified by executing the same sequences through HIP-on-CUDA
   (`hipnv_ts_calls`). Nothing raises any more — see Phase 3 below.
3. ~~Finish Phase 2~~ — dispatch, recording and local arrays are all done, and every
   Dr.Jit-Core suite passes against HIP. What is left of Phase 2 is fp16 atomics and
   the ray-tracing opcodes, the latter belonging with the Phase 0b spike below.
4. ~~Phase 0b spike~~ — HIP-RT traversal COMPILES AND LINKS for gfx90a, and RUNS
   CORRECTLY on NVIDIA (`tools/hip_validate/hiprt_triangle.cpp`: BVH built on device,
   hit at t=1.0 on the inside ray, miss on the outside one). The stock nixpkgs HIP-RT
   cannot target NVIDIA, but that is packaging, not HIP-RT -- `flake.nix` now builds a
   CUDA-enabled `hipRtNv` alongside it. Because gfx90a has NO ray-tracing hardware and
   takes HIP-RT's RTIP 0 SOFTWARE path, NVIDIA runs very nearly the same code the MI210
   will, so this is an unusually good cross-vendor proxy
   (BACKEND_NOTES §7a-b). Three findings change the plan: every traversing kernel must
   define `intersectFunc`/`filterFunc` or the link fails; `hiprtHit` carries only six
   of Metal's eight outputs (no geometry ID, no user instance ID), so §7's "adopt the
   signature verbatim" needs qualifying; and traversal costs 64 VGPR / 38 AGPR / 800 B
   scratch on gfx90a -- `hip_validate` now prints these for every kernel.
5. ~~Phase 3~~ — the device library builds for gfx90a, is embedded, and drives all six
   previously-raising `HIPThreadState` methods. Two width-32 checks say the port is
   algorithmically intact (BACKEND_NOTES §11g); wave64 semantics are the milestone that
   remains, and they are on the MI210 list with the rest.
6. Phases 4–8.

**Still unanswered, and still worth answering:** §8.0, the task-shape question. It does
not block the work above, but it determines whether the phone fleet ends up the primary
platform (§0.1), and it has been open since the start.

**Cheap, parallel, no fleet required (§0):**
5. **Answer the three PanVK gates** (§3.0.1) from existing PanVK work —
   `VK_KHR_buffer_device_address` first. Also: does RADV cover `gfx90a` for compute?
6. **If any single aarch64 unit exists**, measure `llvm_ad_spectral` throughput on it
   (§0.1). A day's work; sizes the fleet question six months early.

**Standing:**
7. **Keep the §3.1 emit layer honest.** It is what makes HIP-now non-wasteful if the
   month-6 decision goes to SPIR-V. Review it at each phase boundary, not at the end.
