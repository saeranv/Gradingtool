# Enable Python support and load DesignScript library
import clr
clr.AddReference('ProtoGeometry')
from Autodesk.DesignScript.Geometry import *
clr.AddReference('RevitAPI') #import clr
from Autodesk.Revit.DB import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append(r"C:\Program Files (x86)\IronPython 2.7\Lib")
sys.path = sorted(sys.path, key=lambda p: p.find("Python27"))

import math
import uuid
import cPickle
import zipfile
import os
import subprocess
import getpass
import System

try:
    System.Net.ServicePointManager.SecurityProtocol = System.Net.SecurityProtocolType.Tls12
except AttributeError:
    # TLS 1.2 not provided by MacOS .NET Core; revert to using TLS 1.0
    System.Net.ServicePointManager.SecurityProtocol = System.Net.SecurityProtocolType.Tls

def download_file(url, targetDirectory):
    localFilePath = os.path.join(targetDirectory, os.path.basename(url))
    client = System.Net.WebClient()
    client.DownloadFile(url, localFilePath)
    return localFilePath

def check_or_make_directory(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except:
            raise Exception("Folder can't be constructed: '{}' can't be made.".format(dir_path))

def unzip_file(zipFile, targetDirectory):
    #unzip the file
    with zipfile.ZipFile(zipFile) as zf:
        for f in zf.namelist():
            if f.endswith('/'):
                try: os.makedirs(f)
                except: pass
            else:
                zf.extract(f, targetDirectory)

def del_directory(dir, rmdir = True):
    try:
        if dir[-1] == os.sep: dir = dir[:-1]
        files = os.listdir(dir)
        for file in files:
            if file == '.' or file == '..': continue
            path = dir + os.sep + file
            if os.path.isdir(path):
                del_directory(path)
            else:
                os.unlink(path)
        if rmdir: os.rmdir(dir)
    except:
        print "Delete old directory failed"


def set_directory_structure(update_tool, cache_dir_parent, pkg_name, url):
    """
    Makes directory to store cache files if it doesn't exist.
    Copies python package over from git.
    """

    # cache directory
    cache_dir = r"kt_cpython_cache"
    cache_path = os.path.join(cache_dir_parent, cache_dir)

    # pkg directory
    pkg_parent_path = os.path.join(cache_path, pkg_name)
    pkg_path = os.path.join(pkg_parent_path, pkg_name + "-master")


    # if package exists
    if os.path.isdir(pkg_path):
        # delete pkg_path if you want to update
        if update_tool == True:
            del_directory(pkg_parent_path, rmdir = True)
            print "Deleting ", pkg_path

    # Check again if package doesn't exists, download from repo
    if not os.path.isdir(pkg_path):
            # cache directory
            check_or_make_directory(cache_path)
            # pacakge parent directory
            check_or_make_directory(pkg_parent_path)

            # download pkg zip
            zip_file_path = download_file(url, pkg_parent_path)
            print "Downloading the source code..."

            # unzip pkg
            unzip_file(zip_file_path, pkg_parent_path)

            print "Downloaded!"

    #run_bat_file(pkg_path)

    return pkg_path

def run_bat_file(directory = "C:/"):
    directory = 'C:/'
    batfpath = os.path.join(directory, 'python_packages.bat')
    with open(batfpath, 'w') as OPATH:
        OPATH.writelines(['conda install numpy\n','y\n','pause'])

    run_python_process(batfpath, directory)


def run_python_process(p, program, working_directory, argument = "", argstr = ""):
    #print program
    #print argument
    #print working_directory
    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    p = subprocess.Popen([program, argument, argstr], cwd=working_directory, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    #print p.communicate()
    script_output = p.stdout.read()
    print script_output


def main(anaconda_path, update_tool = False):
    if anaconda_path is None:
        user_name = getpass.getuser()
        anaconda_path = "C:\\Users\\{}\\AppData\\Local\\Continuum\\miniconda2\\".format(user_name)

    if not os.path.isdir(anaconda_path):
        raise Exception("Cannot find path to Anaconda. It is usually found in C:\Users\%USERNAME%\AppData\Local\Continuum\miniconda2. Try inputting it manually")
        return

    # Find/make all these
    program = os.path.join(anaconda_path, "python.exe")
    pkg_name = "Gradingtool"
    pkg_url = "https://github.com/saeranv/{}/archive/master.zip".format(pkg_name)

    cache_dir_parent = os.path.join(anaconda_path, "pkgs")
    cache_gh_pklname = "listen_app.pkl"
    cache_app_pklname = "listen_gh.pkl"

    package_path = set_directory_structure(update_tool, cache_dir_parent, pkg_name, pkg_url)
    pkg_cache_dir = os.path.join(package_path, "cache")


    return 'Success'

anaconda_path_ = IN[1]
update_ = IN[0]

if IN[0] == True:
    if anaconda_path_ == "": anaconda_path_ = None
    result = main(anaconda_path_, update_tool = update_)
    OUT = result