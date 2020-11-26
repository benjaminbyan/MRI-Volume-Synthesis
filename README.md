# MRI-Volume-Synthesis

This code details a novel three-dimensional MRI volume synthesis system that enables accurate, memory-efficient, and graphically precise translation from one MRI domain to another, including T1-weighted, T2-weighted, T1 contrast, and T2 fluid attenuated inverse recovery (FLAIR) imaging. The model remixes the Pix2Pix architecture for image translation using cycle-consistency loss metrics and bidirectional loss and trianing. In other words, the model simultaneously and synchronously learns the forward mapping of 3D images from domain D1 to D2 as well as the inverse mapping from domain D2 to D1. This has a multitude and diversity of applications for medical imaging, oncological screening, and neurological disease diagnosis, treatment, and therapy. The AI architecture has the potential to substntially reduce the operational cost of MRI imaging, decrease the requisite time for diagnosis, as well as expand the clinical availability of MRI resources. 

![Image of Yaktocat](https://cdn.vox-cdn.com/thumbor/SPV80kJzpBanxbIymi4UGrOHOGY=/0x0:1920x1080/1720x0/filters:focal(0x0:1920x1080):format(webp):no_upscale()/cdn.vox-cdn.com/uploads/chorus_asset/file/21766684/AI_Blog_fastMRI_Desktop.jpg)

The repository includes the code used for engineering and training the generative adverarial network (GAN) model, as well as evaluating loss metrics and building data visualizations such as the three-dimensional reconstruction from 2D axial slices. The data is provided by the Perelman School of Medicine, which is detailed in more information at the following link: https://www.med.upenn.edu/sbia/brats2018/data.html. 
