from utils.pose_utils import gen_poses
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "--working_dir", required=True, type=str, help="Path to generate pose files"
    )
args = parser.parse_args()

if __name__=='__main__':
    gen_poses(args.working_dir)