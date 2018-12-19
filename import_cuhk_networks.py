from shutil import copyfile
import os


def do(origin="AtoB"):
    path1 = os.path.join("checkpoints", "cuhk_pix2pix_" + origin, "latest_net_G.pth")
    if origin == "AtoB":
        target = "latest_net_G_A.pth"
    else:
        target = "latest_net_G_B.pth"

    path2 = os.path.join("checkpoints", "flickr", target)
    copyfile(path1, path2)


if __name__ == "__main__":
    do(origin="AtoB")
    do(origin="BtoA")
