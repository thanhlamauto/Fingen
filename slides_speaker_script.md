# Speaker Script for `slides.tex`

Target: 15 minutes for the main deck. Backup slides are only for Q&A.

## Slide 1 - Title
Good morning/afternoon. Our project is about lightweight cross-sensor fingerprint generation using ridge-conditioned diffusion. The core idea is to split the generation problem into two parts: first generate a new fingerprint identity, then render that identity into a target sensor style using explicit ridge control and a small LoRA adapter.

## Slide 2 - Team Contribution
The three members contributed equally, but we divided the work into three balanced tracks. Lam focused on problem framing, related work, Stage 1, and ridge extraction. Bang focused on Stage 2, DB1-DB4 LoRA transfer, pose-mask augmentation, and the live demo. An focused on recognition evaluation, LightGlue and t-SNE diagnostics, and report/slide consolidation. The final method choices, result checks, and report revisions were reviewed jointly, so we mark the work as equal contribution.

## Slide 3 - Talk Map
I will organize the talk into five parts. First, I will introduce the target-sensor data problem. Then I will briefly cover related work and competitor methods. After that, I will explain our two-stage ridge-conditioned method, followed by the SD302A and FVC 2004 experiments. I will end with ablations, limitations, and the main takeaways.

## Slide 4 - Problem
The practical problem is that fingerprint recognizers need many identities and repeated impressions per identity, but biometric data is expensive, privacy-sensitive, and difficult to share. When a new sensor is deployed, we may only have a small calibration set. Synthetic data therefore has to do two things at the same time: preserve ridge and minutiae evidence for identity, while changing pose, mask, contrast, and artifacts to match the target sensor.

## Slide 5 - Related Work
Previous work can be grouped into three directions. Classical methods such as SFinGe use hand-designed ridge flow and noise models. GAN-based methods such as PrintsGAN and FPGAN-Control learn to generate many synthetic identities and impressions. Diffusion and control methods use latent diffusion, ControlNet-style conditioning, and adapters to control structure or style. Our method combines ideas from these directions, but keeps the model compact and fingerprint-specific.

## Slide 6 - Competitor Methods
Here, the important point is how each competitor generates fingerprints. PrintsGAN uses a GAN to produce synthetic fingers and repeated impressions. FPGAN-Control disentangles identity from appearance and is our closest SD302A baseline. GenPrint uses a Stable-Diffusion-style model with text and image conditions, which is powerful but much larger. Ridge Scale Aligned Diffusion is a useful reference for ridge/style diagnostics. Our difference is that we do not fine-tune a large backbone; we freeze the structural path and train only a small Stage-2 sensor LoRA.

## Slide 7 - Key Idea
The design principle is to separate what must stay fixed from what should change. Identity is mainly represented by ridge flow, ridge frequency, and minutiae layout. Sensor domain is represented by foreground footprint, gray tone, background, pressure, crop, and artifacts. Our contributions are a two-stage pipeline, explicit ridge conditioning, small sensor LoRA adapters, and recognition, LightGlue, t-SNE, and ablation diagnostics.

## Slide 8 - Method Overview
This figure shows the full pipeline. Stage 1 takes noise and generates a rolled fingerprint identity. We then extract a ridge map by binarization. This ridge map is warped and clipped into a pose mask from the target sensor. Finally, Stage 2 renders the image using the corresponding sensor LoRA checkpoint, for example DB1 LoRA. This allows one identity to produce multiple instances under different masks and poses.

## Slide 9 - Stage 1
Stage 1 works in the VQ-VAE latent space, compressing a 512 by 512 image into a 128 by 128 by 3 latent grid. It is an unconditional identity prior, meaning that we do not provide a sensor label or reference image. The Stage-1 output is not the final recognizer training sample; it is an identity source used for ridge extraction.

## Slide 10 - Ridge Condition
After Stage 1, we use Sauvola adaptive binarization to obtain a binary ridge map. Since fingerprints are dominated by ridge structure, this condition is a simple and deterministic way to preserve identity evidence. Instead of letting the renderer hallucinate ridge structure freely, the explicit ridge condition forces Stage 2 to follow the generated identity.

## Slide 11 - Pose-Mask Alignment
To create multiple impressions, we sample pose masks from a target-sensor mask library. The ridge map is aligned using a bounded similarity transform and then clipped into the foreground mask. We also apply mild control augmentation such as rotation, translation, dropout, and speckle noise. This is the component that creates impression-level variation.

## Slide 12 - Stage 2 and LoRA
Stage 2 is the renderer that maps the ridge condition to a sensor-style image. The large components are frozen: VQ-VAE, Stage-1 prior, the ControlNet structural path, and the base weights. For a new sensor, the only trainable part is LoRA in the middle self-attention qkv and output projection layers of the Stage-2 UNet. With rank 8, each sensor has only about 12.3K LoRA parameters.

## Slide 13 - LoRA Rank
The LoRA rank controls the capacity of the adapter. The parameter count grows linearly with rank, and for our selected location it is P(k) = 1536k. We ran a quick sweep with ranks 1, 2, 4, and 8 for DB1 and DB2. Here, EER means equal-error rate: the threshold where false accepts and false rejects are equal, so lower is better. In this reduced setting, rank 4 gives the lowest EER for both databases. However, we do not claim rank 4 is always optimal for every sensor; rank should be selected per sensor using validation EER and structural QC.

## Slide 14 - Experimental Protocol
We use two main protocols. For SD302A, we compare synthetic-only and real-plus-synthetic training using ResNet18, ResNet50, and ViT recognizers. For FVC 2004, each database uses only 80 DBx_A images to fit the LoRA adapter, while DBx_B is kept for evaluation. This setup tests whether the generator can transfer to a new sensor style with limited data.

## Slide 15 - SD302A Result
This table uses TAR at FAR 0.1 percent. That means we first fix a strict false-accept rate of 0.1 percent, then measure how many genuine pairs are still correctly accepted. Higher is better. Under this metric, our synthetic data is competitive with the strongest baseline in the synthetic-only setting. When combined with real data, it improves ResNet18 and ViT performance clearly. The main message is not that synthetic data completely replaces real data, but that it is useful augmentation when the real training set is limited.

## Slide 16 - SD302A to FVC DB1 Transfer
This figure illustrates the transfer process. The ridge can come from SD302A or from a Stage-1 synthetic identity, but when rendering to DB1, we choose a DB1 pose mask, warp and clip the ridge into that mask, and sample Stage 2 using the DB1 LoRA. Therefore, the final sensor style is controlled by the target mask and the DB1 adapter.

## Slide 17 - FVC Recognition
For FVC 2004, we again report TAR at FAR 0.1 percent, so all databases are compared under the same strict false-accept constraint. Real plus synthetic improves all four databases on DBx_B evaluation. Synthetic-only is also competitive for DB1 and DB3, but DB2 is harder because it has weaker ridge/valley contrast. This shows that the small adapter is useful, but the quality of the target sensor still has a strong effect.

## Slide 18 - LightGlue Diagnostic
LightGlue is not used here as a biometric matcher. It is a diagnostic to check whether the pose-aligned ridge hint and the Stage-2 output still share local structure. The match results are reasonably strong, especially for DB3, which suggests that the renderer does not completely destroy the ridge condition.

## Slide 19 - t-SNE Diagnostic
The t-SNE uses simple descriptors based on foreground shape, intensity, contrast, and thumbnails. The goal is to check whether DB-specific LoRA adapters produce different styles. The silhouette score of 0.446 suggests that the adapters learn distinct domain styles. DB1 separates most clearly, while DB2 and DB4 are closer.

## Slide 20 - Ablations
The most important ablations are latent resolution and ridge extraction. The 128 by 128 latent grid preserves ridge topology much better than the 32 by 32 proxy. Sauvola also provides denser control support than Canny, because Canny mainly captures contours and fragmented edges. The DB1/DB2 rank sweep is the new ablation for adapter capacity.

## Slide 21 - Efficiency
The practical advantage is model size. The full two-stage generator has about 53.58M parameters, the active Stage-2 path has 23.39M parameters, and each rank-8 sensor LoRA has only 12.3K parameters. Compared with a Stable-Diffusion-style backbone, this is a much lighter option for smaller labs or limited-GPU environments.

## Slide 22 - Limitations
There are three main limitations. First, Stage 2 depends on ridge extraction and pose-mask alignment; if warping or clipping is wrong, minutiae-level information can be lost. Second, the rank sweep is still preliminary and only covers DB1 and DB2, so it is not enough to conclude a universal optimal rank. Third, LightGlue and t-SNE are supporting diagnostics, not replacements for full biometric validation.

## Slide 23 - Takeaways
To summarize, RidgeLoRA-FP separates identity generation from sensor rendering. Ridge conditioning preserves identity evidence, masks and poses create impression variation, and a small LoRA adapter enables transfer to a new sensor. The results show that real plus synthetic data improves recognition on SD302A and FVC, while low-quality sensors such as DB2 remain challenging.

## Backup Slides
Use these only for Q&A.

Backup Sauvola vs. Canny: Sauvola preserves dense ridge-phase support, while Canny tends to fragment the structure.

Backup LoRA Placement: LoRA is applied only to Stage-2 middle attention qkv/proj, not to ControlNet, because we do not want to modify the structural ridge path.

Backup DB2 Diagnostic: DB2 has lower contrast, so synthetic-only training is harder and the small adapter cannot fully compensate for the quality issue.

Backup Real vs. Synthetic Pages: use these pages to show that each DB has a distinct style and that the synthetic images are not simply copied real samples.
