# ECGSEG
## Task: Combine computer vision with sequencial analysis
Given periodic video data (yearly medical scan, daily survance video), provide classification and inference(heart attack % in next 5 years, existence of a crime).

Alot of difficulty is in managing computational costs. Video classification models are already very intensive (3D resnet), how to preform sequencial video classification?

Here, we use a repeated feature extraction model in conjunction with a sequencial inference model.

The kinitic action identification dataset had each video segmented into five pieces to simulate preforming sequencial video classification. 

Still, optimization is slow and difficult without an imagenet-equivolent transfer base. 

#### to use, download ucf kinitics data into its data directory, then run data utils to generate pytorch dataset
