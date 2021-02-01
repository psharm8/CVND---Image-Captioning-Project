import os
import sys
import zipfile
import urllib.request
import shutil

from tqdm import tqdm

pbar = None
blocksDone = 0


def progress(count, blockSize, totalSize):
    global pbar
    global blocksDone

    if pbar is None:
        pbar = tqdm(total=totalSize)

    delta = count - blocksDone
    blocksDone = count

    pbar.update(blockSize * delta)
    currentSize = count * blockSize
    if currentSize >= totalSize:
        pbar.close()
        pbar = None
        blocksDone = 0


def fetchZip(srcUrl: str, dest: str):
    if not os.path.exists(dest):
        print(f"Downloading zip to {dest}")
        urllib.request.urlretrieve(srcUrl, dest, progress)


def unzip(zFile: str, dest: str):
    print(f"Extracting {zFile}")
    with zipfile.ZipFile(zFile, "r") as z:
        for file in tqdm(z.infolist()):
            try:
                z.extract(file, dest)
            except zipfile.error:
                pass
    print()

def getDataSet(version: str, destDir='data'):
    cocoDir = os.path.join(destDir, "cocoapi")
    imagesDest = os.path.join(cocoDir, "images")
    annotationsDest = os.path.join(cocoDir, "annotations")

    if not os.path.exists(annotationsDest):
        os.makedirs(annotationsDest)
    if not os.path.exists(imagesDest):
        os.makedirs(imagesDest)

    annUrl = f"http://images.cocodataset.org/annotations/annotations_trainval{version}.zip"
    annFile = f"{destDir}/annotations_trainval{version}.zip"

    imInfoUrl = f"http://images.cocodataset.org/annotations/image_info_test{version}.zip"
    imInfoFile = f"{destDir}/image_info_test{version}.zip"

    imTrainUrl = f"http://images.cocodataset.org/zips/train{version}.zip"
    imTrainFile = f"{destDir}/train{version}.zip"

    imValUrl = f"http://images.cocodataset.org/zips/val{version}.zip"
    imValFile = f"{destDir}/val{version}.zip"

    imTestUrl = f"http://images.cocodataset.org/zips/test{version}.zip"
    imTestFile = f"{destDir}/test{version}.zip"

    if not os.path.exists(f"{annotationsDest}/.check"):
        fetchZip(annUrl, annFile)
        unzip(annFile, cocoDir)
        fetchZip(imInfoUrl, imInfoFile)
        unzip(imInfoFile, cocoDir)
        with open(os.path.join(annotationsDest, ".check"), "w") as check:
            check.write("ok")

    imTrainDest = os.path.join(imagesDest, f"train{version}")
    imValDest = os.path.join(imagesDest, f"val{version}")
    imTestDest = os.path.join(imagesDest, f"test{version}")

    if not os.path.exists(f"{imTrainDest}/.check"):
        fetchZip(imTrainUrl, imTrainFile)
        unzip(imTrainFile, imagesDest)
        with open(os.path.join(imTrainDest, ".check"), "w") as check:
            check.write("ok")

    if not os.path.exists(f"{imValDest}/.check"):
        fetchZip(imValUrl, imValFile)
        unzip(imValFile, imagesDest)
        with open(os.path.join(imValDest, ".check"), "w") as check:
            check.write("ok")

    if not os.path.exists(f"{imTestDest}/.check"):
        fetchZip(imTestUrl, imTestFile)
        unzip(imTestFile, imagesDest)
        with open(os.path.join(imTestDest, ".check"), "w") as check:
            check.write("ok")
