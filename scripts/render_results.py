import os
import torch
import argparse
import numpy as np
import smplx as SMPL
from PIL import Image
from pytorch3d.io import load_obj
from utils.smpl_helpers import render_360_gif


class Render360:

    def __init__(self, pose_id) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.pose_id = pose_id

        #   creates a SMPL instance, and samples it T pose
        smpl_path = os.path.abspath(os.path.join(__file__ ,"../../sample-data/SMPL/models/"))
        print(smpl_path)
        self.smpl = SMPL.create(smpl_path, model_type='smpl', gender='male')

        self.body_pose = torch.zeros(1,69)
        self.betas = torch.zeros(1,10)
        self.adjust_body_pose(pose_id)
        self.smpl_output = self.smpl( betas=self.betas,
                            body_pose=self.body_pose,
                            return_verts=True)

        self.verts = self.smpl_output.vertices[0]
        self.faces = self.smpl.faces_tensor

        #   loads the SMPL template mesh to extract UV coordinates for the textures
        mesh_filename = os.path.join(__file__,"../../sample-data/smpl_uv_20200910/smpl_uv.obj")
        _, self.faces_verts, aux = load_obj(mesh_filename)
        self.verts_uvs = aux.verts_uvs[None, ...]        # (1, F, 3)
        self.faces_uvs = self.faces_verts.textures_idx[None, ...] 

    def adjust_body_pose(self, pose_id):
        if pose_id == 1:
            self.body_pose[0, 0] = -2  # LEFT LEG Back_Forward : Moves the leg back or forward, positive for back and negative for forward
            self.body_pose[0, 3] = -1.6 # RIGHT LEG Back_Forward: Moves the leg back or forward, postiive for back and negative for forward
            self.body_pose[0, 9] = 0.3 # LEFT LEG X axis knee bend: positive for back, and negative for forward
            self.body_pose[0, 11] = -1.5 # LEFT LEG Z axis knee rotation sideways: positive for outward, negative for inward
            self.body_pose[0, 12] = 1.6 # RIGHT LEG X axis knee bend: positive for back and negative for forward (negative is not recommended)
            self.body_pose[0, 18] = 0.9 # LEFT ANKLE X axis (do ballet move or the move you do to look taller): positive for down, negative for up
            self.body_pose[0, 21] = -0.1 # RIGHT ANKLE X axis (do ballet move): positive for down, and negative for up
            self.body_pose[0, 45] = -0.5 # LEFT SHOULDER JOINT WITHOUT SHOLDER ROTATE X axis: positive for inward, negative for outward
            self.body_pose[0, 47] = -0.7 # LEFT SHOULDER JOINT WITHOUT SHOULDER ROTATE Z axis up/ down: positive for up and negative for down
            self.body_pose[0, 48] = -1.2 # RIGHT SHOULDER JOINT WITHOUT SHOULDER (FOREARM?) ROTATE X axis inward /outward: positive for inward, negative for outward
            self.body_pose[0, 50] = 1.4 # RIGHT SHOULDER JOINT WITHOUT SHOULDER up / down Rotate Z axis : positive for down, negative for up
            self.body_pose[0, 53] = -1.7 # LEFT HAND FORARM ROTATE Z axis up down : positive for up and negative for down
            self.body_pose[0, 54] = 1.2 # RIGHT FOREARM ROTATE X axis : positive for inward, negative for outward
            self.body_pose[0, 55] = 0.1 # RIGHT FOREARM ROTATE Y axis back / forward : positive for inward (forward), negative for outward (backward)
            self.body_pose[0, 56] = 0.1 # RIGH FORARM ROTATE Z axis up / down : positive for down and negative for up
        elif pose_id == 2:
            self.body_pose[0, 47] = -1.5 # LEFT SHOULDER JOINT WITHOUT SHOULDER ROTATE Z axis up/ down: positive for up and negative for down
            self.body_pose[0, 50] = 1.5 # RIGHT SHOULDER JOINT WITHOUT SHOULDER up / down Rotate Z axis : positive for down, negative for up
        elif pose_id == 3:
            self.body_pose[0, 0] = 0.2  # LEFT LEG Back_Forward : Moves the leg back or forward, positive for back and negative for forward
            self.body_pose[0, 47] = -0.70 # LEFT SHOULDER JOINT WITHOUT SHOULDER ROTATE Z axis up/ down: positive for up and negative for down
            self.body_pose[0, 50] = 1.5 # RIGHT SHOULDER JOINT WITHOUT SHOULDER up / down Rotate Z axis : positive for down, negative for up
            self.body_pose[0, 53] = -1.5 # LEFT HAND FORARM ROTATE Z axis up down : positive for up and negative for down
            self.body_pose[0, 57] = 1.5 # LEFT PALM ROTATE X axis : positive for inward and negative for outward
            self.body_pose[0, 58] = -1 # LEFT PALM ROTATE Y axis : positive for back, negative for forward
            self.body_pose[0, 59] = 0 # LEFT PALM ROTATE Z axis : positive for up and negative for down
            self.body_pose[0, 45] = -0.3 # LEFT SHOULDER JOINT WITHOUT SHOLDER ROTATE X axis: positive for inward, negative for outward

    def render_textures(self, textures_folder, output_dir):

        #   extract list of texture files
        if os.path.exists(textures_folder):
            files = os.listdir(textures_folder)
        else:
            print("ERROR: ", textures_folder, " does not exit")

        for idx, current_file in enumerate(files):
            current_texture_path = os.path.join(textures_folder, current_file)
            output_texture_path = os.path.join(output_dir, current_file)
            print('\nProcessing image ', current_texture_path)

            if ".jpg" in current_texture_path or ".png" in current_texture_path:
                with Image.open(current_texture_path) as image:
                    current_image_np = np.asarray(image.convert("RGB")).astype(np.float32)

                render_360_gif(self.device, self.verts,
                        current_image_np, self.verts_uvs,
                        self.faces_uvs, self.faces_verts.verts_idx,
                        output_texture_path.replace(".png", f"-POSEID-{self.pose_id}-360.gif"))
                break

parser = argparse.ArgumentParser(description= 'Renders SMPL 360 gifs given input textures')
parser.add_argument('--textures', type=str, help='Folder with textures', required=True)
parser.add_argument('--resultDir', type=str, help='Output folder for 360 gifs', required=True)
parser.add_argument('--pose_id', type=int, help='Pose ID', required=True)

args = parser.parse_args()

INPUT_FOLDER = args.textures
OUTPUT_FOLDER = args.resultDir
POSE_ID = args.pose_id

render = Render360(POSE_ID)
render.render_textures(INPUT_FOLDER, OUTPUT_FOLDER)