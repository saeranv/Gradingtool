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
    print targetDirectory
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
        if update_tool:
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
    return pkg_path

def run_python_process(p, program, argument, working_directory):
    argstr = ""
    print program
    print argument
    print working_directory
    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    p = subprocess.Popen([program, argument, argstr], cwd=working_directory, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    #print p.communicate()
    script_output = p.stdout.read()
    streamdata = p.communicate()[0] # need this to make sure process has finished
    exitcode = p.returncode

    if exitcode != 0:
        print "Command failed with return code", exitcode

    #script_output = program + argument + working_directory

    return p, script_output, exitcode

def send_to_pickle(znames_, aval_dict_, apts_mtx_, exp_, param_, cache_dir_):

    exp_run_name = experiment_run_name(exp_, param_)

    for i,apts_in_zone in enumerate(apts_mtx_):

        for j,apt in enumerate(apts_in_zone):
            apt = rs.coerce3dpoint(apt)
            apts_mtx_[i][j] = [apt[0],apt[1],apt[2]]

    #print(apts_mtx_[-1])
    #print(len(apts_mtx_[-1]))

    pkl_mtx = [znames_, apts_mtx_, aval_dict_]

    return pickle(pkl_mtx, cache_dir_, exp_run_name + ".pkl")

def pickle(data, cpickle_path, cpickle_name):
    pkl_file_path = os.path.join(cpickle_path, cpickle_name)

    # Pickle objects, protocol 1 b/c binary file
    with open(pkl_file_path, "wb") as outf:
        cPickle.dump(data, outf)

    return pkl_file_path

def unpickle(pkl_file_path):
    """ Read non-binary picle """
    with open(pkl_file_path, "r") as inf:
        unpickled_object = cPickle.load(inf)
        return unpickled_object

def rvt2py(rhpt):
    if isinstance(rhpt, (list,tuple)):
        rhpt_lst = rhpt
        return [(rhpt.X, rhpt.Y, rhpt.Z) for rhpt in rhpt_lst]
    else:
        return (rhpt.X, rhpt.Y, rhpt.Z)

def py2rvt(pypt):
    if isinstance(pypt, (list,tuple)):
        pypt_lst = pypt
        return [XYZ(pypt[0],pypt[1],pypt[2]) for pypt in pypt_lst]
    else:
        return XYZ(pypt[0],pypt[1],pypt[2])


def main(p, topo_pts, topo_crv, bldg_crv, anaconda_path, perpendicular_dist, spacing_dist, update_tool = False):
    msg = "Script didn't run."
    boundpts = None
    number = None


    if not os.path.isdir(anaconda_path):
        user_name = getpass.getuser()
        anaconda_path = "C:\\Users\\{}\\AppData\\Local\\Continuum\\miniconda2\\".format(user_name)

    if not os.path.isdir(anaconda_path):
        where_path = "Cannot find path to mininaconda at {}.".format(anaconda_path)
        raise Exception( where_path +
                        " It is usually found in C:\Users\%USERNAME%\AppData\Local\Continuum\miniconda2. "
                        "Try inputting it manually")


    # Find/make all these
    program = os.path.join(anaconda_path, "python.exe")
    pkg_name = "Gradingtool"
    pkg_url = "https://github.com/saeranv/{}/archive/master.zip".format(pkg_name)
    
    cache_dir_parent = os.path.join(anaconda_path, "pkgs")
    cache_gh_pklname = "listen_app.pkl"
    cache_app_pklname = "listen_gh.pkl"
    
    package_path = set_directory_structure(update_tool, cache_dir_parent, pkg_name, pkg_url)
    pkg_cache_dir = os.path.join(package_path, "cache")
    
    
    # Pickle Grasshopper geometry

    bldg_crvpts = rvt2py(bldg_crv)
    topo_crvpts = rvt2py(topo_crv)
    topo_pts = [rvt2py(pt) for pt in topo_pts]

    msg = bldg_crvpts
    D = {}
    D["note"] = "testing from dynamo"
    D["curves"] = bldg_crvpts
    D["points"] = topo_pts
    D["topocrvs"] = topo_crvpts
    D["perpendicular_dist"] = perpendicular_dist
    D["spacing_dist"] = spacing_dist

    cachefpath = pickle(D, pkg_cache_dir, cache_gh_pklname)

    # Run run.py
    argument = os.path.join(package_path, pkg_name.lower(), "run.py")
    working_directory = package_path

    p, std_out, exit_code = run_python_process(p, program, argument, working_directory)
    msg = std_out

    # hack
    if exit_code != 0:
        # hack
        program_anaconda = os.path.join(anaconda_path.replace("miniconda2", "Anaconda2"), "python.exe")
        p, std_out, exit_code = run_python_process(p, program_anaconda, argument, working_directory)
        msg = std_out

    if exit_code != 0:
        return p, None, None, msg

    pkl_file = os.path.join(pkg_cache_dir, cache_app_pklname)
    D = unpickle(pkl_file)
    boundpts = D["closepts"]
    number = D["number"]
    # boundpts = [xyz.ToXyz()for xyz in py2rvt(boundpts)]
    boundpts = py2rvt(boundpts)

    return p, boundpts, number, msg

# Rename inputs
update_ = False
anaconda_path_ = ""
_run = IN[0]
_topo_pts = IN[1]
_topo_crv = IN[2]
_bldg_crv = IN[3]
_perp_dist = IN[4]
_space_dist = IN[5]

p = None

if _run:
    result = main(p, _topo_pts, _topo_crv, _bldg_crv, anaconda_path_, _perp_dist, _space_dist, update_tool = update_)
    p, boundpts, grade_avg, msg = result

    OUT = [msg, grade_avg, boundpts]
else:
    if p!=None:
        p.kill()
