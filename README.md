# IDEA 

Photoneo doesnt work so we will probably just have a bunch of 2D images. 
So we will probably use some neural MVE like in Yairv: Multiview Neural Surface Reconstruction with Implicit Lighting and Material
Questions: 
    How to represent 3D: 
        planes 
            +has no problem with modelling leafs even if they are infinitelly thin 
            -how to make a neural network regress plane parameters
        voxels 
            -will need way to high of a resolution to represent leafs
        Implicit SDF 
            Querying xyz in space gives signed distance
            +allows for doing some arithmetic with the Z code that generated the SDF so that it can be interpolated and so on. This will help in the case of time things
            -probably difficult to model things that are very thin like leafs 
                Ideas for this limitation:  
                    Maybe make the network return two distances one for the front surface and one for the back surfaces together with their normals. 
                    When rendering we choose the surface that has the normal aligned with the view direction. 
Answer: 
    Probably I'll represent it as a implicit SDF conditioned on a Z vector from all the images

full idea: 
    encode each image into a Z vector that is Nx3, which we can rotate and translate into another view and then agregate there another Z which is Nx3. 
        This idea is inspired by  Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation
        It's better than just aggregating by averaging all the Z as it also takes into acount the positions of the camera relative to each other
    In the end we will have a Z which is Nx3 and expressed in one of the cam frames. We can then transform it into the world frame where we will do all computations  
    Now this Z is kind alike the global information 
    At render time we use ray tracing:
        Option 1: 
            Render network receives sample point p in 3D space(possibly with positional encoding), the Z as a Nx3 flattened vector, and the viewing direction v 
            The render network then regresses the SDF value
        Option 2: 
            From this Z we regress the hyperparameters of a Siren and then we use the siren for SDF regression
            Then at render time we pass to the siren network the sample point p and the view direction v to regress the SDF
    At texture time: 
        Option 1:
            Whatever surface intersection we got we pass it through another siren that regresses the RGB texture at that point. We project this to another view and compare pixelwise there.
        Option 2: 
            We map from the surface point to a uv position and sample our color from there
    Add also a loss on the Z vector by using it to regress another image from another pose. This will ensure that it learns a Z meaningful in 3D
    For making the forwards and backward pass fast, use this: DIST: Rendering Deep Implicit Signed Distance Function with Differentiable Sphere Tracing
