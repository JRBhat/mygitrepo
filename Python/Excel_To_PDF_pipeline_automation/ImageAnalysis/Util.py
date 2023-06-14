# $Id: Util.py 12168 2021-04-30 11:58:06Z ndrews $
"""
Util.py

Contains simple/standard Utils

"""
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import map
from builtins import zip
from builtins import next
from builtins import str
from builtins import range
__version__ = "$Revision: 12168 $"


def __svndata__():
    """
    | $Author: ndrews $
    | $Date: 2021-04-30 13:58:06 +0200 (Fr., 30 Apr 2021) $
    | $Rev: 12168 $
    | $URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/Util.py $
    | $Id: Util.py 12168 2021-04-30 11:58:06Z ndrews $
    """
    #only for documentation purpose
    return {
            'author': "$Author: ndrews $".replace('$', '').replace('Author:', '').strip(),
            'date': "$Date: 2021-04-30 13:58:06 +0200 (Fr., 30 Apr 2021) $".replace('$', '').replace('Date:', '').strip(),
            'rev': "$Rev: 12168 $".replace('$', '').replace('Rev:', '').strip(),
            'id': "$Id: Util.py 12168 2021-04-30 11:58:06Z ndrews $".replace('$', '').replace('Id:', '').strip(),
            'url': "$URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/Util.py $".replace('$', '').replace('URL:', '').strip()
            }


import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.info("Importing %s, Vs: %s" % (__name__, __svndata__()['id']))


def __imports__():
    """
    Additional imports

    | Queue(built-in)=2.7.2
    | cPickle=1.71
    | datetime(built-in)=2.7.2
    | fnmatch(built-in)=2.7.2
    | glob(built-in)=2.7.2
    | hashlib(built-in)=2.7.2
    | imp(built-in)=2.7.2
    | itertools(built-in)=2.7.2
    | json=2.0.9
    | jsonpickle=0.4.0
    | psutil=0.4.1
    | queue(built-in)=2.7.2
    | re=2.2.1
    | shutil(built-in)=2.7.2
    | subprocess(built-in)=2.7.2
    | threading(built-in)=2.7.2
    | traceback(built-in)=2.7.2
    """
    #only for documentation purpose
    pass


import os
import glob
import fnmatch
import datetime
import shutil
import sys
import hashlib
import subprocess
try:
    import queue as queue
except ImportError:
    import queue
import threading
import re
import itertools
import json
import jsonpickle
jsonpickle.set_preferred_backend('json')
jsonpickle.set_encoder_options('json', sort_keys=False, indent=4)
import imp
import psutil
import pickle

def grouper(iterable, n, fillvalue=None):
    """helper tool to group list into groups of n 

    :param iterable: input list
    :type iterable: list
    :param n: group size
    :type n: number
    :param fillvalue: fill empty groups with
    :type fillvalue: value(None)
    :return: list of list in groups of n
    :rtype: list of list
    """
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_python_processes(lst, PROCNAME="python.exe"):
    """gets all python processes from an list as generator

    :param lst: collection of processes
    :type lst: Iterable of processes 
    :param PROCNAME: processname of python, defaults to "python.exe"
    :type PROCNAME: str, optional
    :yield: processes
    :rtype: generator
    """
    for proc in lst:
        try:
            if proc.name == PROCNAME:
                yield proc
        except Exception:
            pass

def get_python_memory():
    """gets the sum of memory used by python processes

    :return: sum of memory used by python processes
    :rtype: [type]
    """
    return sum([p.get_memory_info().rss for p in get_python_processes(psutil.process_iter(), PROCNAME="python.exe")])

def multi_delete(list_, *args):
    """deletes objects from list, pass addtional list with indexes to delet

    :param list_: list of objects to delet
    :type list_: object
    :return: list without deleted objects
    :rtype: list
    """
    try:
        try:
            if len(args) == 1:
                indexes = args
        except:
            indexes = sorted(list(args), reverse=True)
        for index in indexes:
            if isinstance(index, (list, tuple)):
                index = index[0]
            del list_[index]
    except:
        pass
    return list_


def unshared_copy(inList):
    """creates an unshared copy of a list

    :param inList: list to copy
    :type inList: list
    :return: unshared copy if inList is List else original list
    :rtype: List
    """
    if isinstance(inList, list):
        return list(map(unshared_copy, inList))
    return inList


def map_round_list(inList):
    """if a list is passed creates an unshared copy of a list else if a number is passed rounds it

    :param inList: list to copy or number to round
    :type inList: list/numver
    :return: unshared copy if inList is List else if inList is number rounded number
    :rtype: List
    """

    if isinstance(inList, list):
        return list(map(map_round_list, inList))
    return round(inList)


def delFileShell(file):
    """creates a new process which tries to delet a file as administrator

    :param file: file do delet
    :type file: string
    """
    try:
        subprocess.call(['runas', '/user:Administrator', 'del "' + file + '"'])
    except Exception as inst:
        LOGGER.error("Could not remove existing destination File '%s', Exception: %s" % (file, inst)) #log error
        sys.exit(1)

def create_filename_from_basefile(filename, file_ext=None, directory=None, file_prep=""):
    """creates a new file loaction from given filename

    :param filename: filename of the original file
    :type filename: str
    :param file_ext: new file extension, defaults to None
    :type file_ext: str, optional
    :param directory: directory of the new file, defaults to None
    :type directory: new directory, optional
    :param file_prep: file prefix, defaults to ""
    :type file_prep: str, optional
    :return: a new filelocation string
    :rtype: string
    """
    basename, ext = os.path.splitext(filename)
    if directory is None:
        directory, basename = os.path.split(basename)
    else:
        _, basename = os.path.split(basename)
    if file_ext is None:
        file_ext = ext
    if file_prep is None:
        file_prep = ""
    outFile = os.path.join(directory, file_prep + basename + file_ext)
    return outFile


def getPathDepth(inpath):
    """returns the path depth of an path

    :param inpath: path
    :type inpath: string
    :return: depth of path
    :rtype: int
    """
    return len(os.path.normpath(inpath).split(os.sep))

def get_basefilename(filename):
    """gets the filename without path and extension

    :param filename: filepath
    :type filename: str
    :return: basfilename
    :rtype: str
    """
    return os.path.splitext(os.path.split(filename)[-1])[0]

    """ finds files by mask in path and subdirectories (until depth)
    :param path: starting path
    :type path: string
    :param fileMask: standard filemask with *,? etc
    :type fileMask: string
    :param depth: standard filemask with *,? etc
    :param depth: 
    :type depth: type(0)
    :rtype: list of files whole """
def getAllFiles(inpath, fileMask, depth=-1, casesensitiv=False):
    """ finds files by mask in path and subdirectories (until depth)

    :param inpath: starting path
    :type inpath: str
    :param fileMask: standard file mask
    :type fileMask: str
    :param depth: max depth to search (inclusiv), defaults to -1
    :type depth: int, optional
    :param casesensitiv: casesensitiv, defaults to False
    :type casesensitiv: bool, optional
    :return: list of all file paths
    :rtype: list
    """
    matches = []
    if not(isinstance(inpath, (list, tuple))):
        inpath = [inpath]
    if not(isinstance(fileMask, (list, tuple))):
        fileMask = [fileMask]
    for cpath in inpath:
        if depth >= 0: 
            baseDepth = getPathDepth(cpath)
        else: 
            baseDepth = 0
        for fMask in fileMask:
            for root, dirnames, filenames in os.walk(cpath):
                if depth >= 0: 
                    cDepth = getPathDepth(root)
                else: 
                    cDepth = 0
                if (depth < 0) or (cDepth - baseDepth <= depth):
                    if not casesensitiv:
                        for filename in fnmatch.filter(filenames, fMask):
                            matches.append(os.path.normpath(os.path.join(root, filename)))
                    else:
                        matches.extend([os.path.normpath(os.path.join(root,n)) for n in filenames if fnmatch.fnmatch(n, fMask)])
    return matches

def getAllFilesIter(inpath, fileMask, depth=-1, casesensitiv=False):
    """finds files by mask in path and subdirectories (until depth)

    :param inpath: starting path
    :type inpath: str
    :param fileMask: standard file mask
    :type fileMask: str
    :param depth: max depth to search (inclusiv), defaults to -1
    :type depth: int, optional
    :param casesensitiv: casesensitiv, defaults to False
    :type casesensitiv: bool, optional
    :yield: returns a generator with all found files
    :rtype: generator
    """
    matches = []
    if not(isinstance(inpath, (list, tuple))):
        inpath = [inpath]
    if not(isinstance(fileMask, (list, tuple))):
        fileMask = [fileMask]
    for cpath in inpath:
        if depth >= 0: 
            baseDepth = getPathDepth(cpath)
        else: 
            baseDepth = 0
        for fMask in fileMask:
            for root, dirnames, filenames in os.walk(cpath):
                if depth >= 0: 
                    cDepth = getPathDepth(root)
                else: 
                    cDepth = 0
                if (depth < 0) or (cDepth - baseDepth <= depth):
                    if not casesensitiv:
                        for filename in fnmatch.filter(filenames, fMask):
                            yield os.path.normpath(os.path.join(root, filename))
                    else:
                        for ff in [os.path.normpath(os.path.join(root,n)) for n in filenames if fnmatch.fnmatch(n, fMask)]:
                            yield ff


def getAllDirs(inpath, fileMask, depth=-1, minDepth=None, casesensitiv=False):
    """finds files by mask in path and subdirectories (until depth)

    :param inpath: list of input path
    :type inpath: [str]
    :param fileMask: list of mask
    :type fileMask: [mask] see https://winscp.net/eng/docs/file_mask
    :param depth: max depth to search (inclusiv), defaults to -1
    :type depth: int, optional
    :param minDepth: min depth, defaults to None
    :type minDepth: int, optional
    :param casesensitiv: casesensitiv, defaults to False
    :type casesensitiv: bool, optional
    :return: list of filepaths
    :rtype: [str]
    """
    matches = []
    if not(isinstance(inpath, (list, tuple))):
        inpath = [inpath]
    if not(isinstance(fileMask, (list, tuple))):
        fileMask = [fileMask]
    for cpath in inpath:
        if depth >= 0: 
            baseDepth = getPathDepth(cpath)
        else: 
            baseDepth = 0
        for fMask in fileMask:
            for root, dirnames, filenames in os.walk(cpath):
                if depth >= 0: 
                    cDepth = getPathDepth(root)
                else: 
                    cDepth = 0
                if (depth < 0) or (cDepth - baseDepth <= depth):
                    if not casesensitiv:
                        for filename in fnmatch.filter(dirnames, fMask):
                            matches.append(os.path.normpath(os.path.join(root, filename)))
                    else:
                        matches.extend([os.path.normpath(os.path.join(root,n)) for n in dirnames if fnmatch.fnmatch(n, fMask)])
    return matches


def createDirectory(dirname):
    """creates directory structure if not existing

    :param dirname: Directorypath
    :type dirname: str
    :return: true if sucess or already existing
    :rtype: bool
    """
    if not os.path.exists(dirname):
        LOGGER.debug("Create path %s" % dirname)
        try:
            os.makedirs(dirname)
        except Exception as inst:
            LOGGER.error("Could nor create directory '%s', Exception: %s" % (dirname, inst)) #log error
            sys.exit(1)
    LOGGER.debug("Path %s exist" % dirname)
    return True

def getModificationTime(filename):
    """get the last modification time as (day, month, year)

    :param filename: file path
    :type filename: str
    :return: (day, month, year)
    :rtype: tupel
    """
    t = os.path.getmtime(filename)
    k = datetime.date.fromtimestamp(t)
    return (k.day, k.month, k.year)

def backupFile(inFile, backupExtension=".bck", maxBackup=-1):
    """creates a backup of an file

    :param inFile: file location
    :type inFile: str
    :param backupExtension: extension of backupfile, defaults to ".bck"
    :type backupExtension: str, optional
    :param maxBackup: how many backups max should be created, defaults to -1
    :type maxBackup: int, optional
    :return: file destination of the last created backup
    :rtype: str
    """
    if not(os.path.exists(inFile)):
        LOGGER.debug("No Backup possible, %s does not exist" % (inFile))
        return 
    backupFileName = inFile + backupExtension
    if maxBackup != 0 :
        ct = 0
        while os.path.exists(backupFileName) and ((maxBackup < 0) or (ct < maxBackup)):
            backupFileName = inFile + ".%d" % ct + backupExtension
            ct += 1
    LOGGER.debug("copy %s -> %s" % (inFile, backupFileName))
    try:
        shutil.copy(inFile, backupFileName)
    except Exception as inst:
        LOGGER.error("Could copy '%s -> %s', Exception: %s" % (inFile, backupFileName, inst)) #log error
        sys.exit(1)
    return backupFileName

def secureMoveFile(inFile, outFile, backupDir=None):
    """copies and moves a file to a new location deleting the old file, verifying equality by using md5 hashes

    :param inFile: file to move
    :type inFile: str
    :param outFile: outfile to copy infile to
    :type outFile: str
    :param backupDir: backup directory to move outfile to if outfile path doesn´t exist , defaults to None
    :type backupDir: str, optional
    """
    if backupDir is None:
        backupDir = os.path.join(os.path.split(outFile)[0])
    if os.path.isdir(outFile):
        outFile = os.path.join(outFile, os.path.split(inFile)[-1])
    if os.path.exists(outFile):
        backupFile(outFile)
        try:
            os.remove(outFile)
        except Exception as inst:
            LOGGER.error("Could not remove existing destination File '%s', Exception: %s" % (outFile, inst)) #log error
            sys.exit(1)
        LOGGER.debug("Copy %s -> %s" % (inFile, outFile))
        shutil.copy(inFile, outFile)
    else:
        createDirectory(backupDir)
        LOGGER.debug("Copy %s -> %s" % (inFile, outFile))
        shutil.copy(inFile, outFile)
    md5_1 = md5_for_file(inFile)
    md5_2 = md5_for_file(outFile)
    LOGGER.debug("Copied %s -> %s, MD5: %s -> %s" % (inFile, outFile, md5_1, md5_2))
    if md5_1 != md5_2:
        LOGGER.error("MD5Error in copying '%s' -> '%s', Exception: %s, %s" % (inFile, outFile, md5_1, md5_2)) #log error
        sys.exit(1)
    else:
        os.remove(inFile)



def secureCopyFile(inFile, outFile, backupDir=None):
    """creates a copy of a infile to outfile, if outfile exists backups outfile first else creates backup directory

    :param inFile: infile to copy
    :type inFile: str
    :param outFile: outfile to copy in
    :type outFile: str
    :param backupDir: backup directory, defaults to None
    :type backupDir: str, optional
    :return: infile
    :rtype: str
    """
    if backupDir is None:
        backupDir = os.path.join(os.path.split(outFile)[0])
    if os.path.exists(outFile):
        backupFile(outFile)
        try:
            os.remove(outFile)
        except Exception as inst:
            LOGGER.error("Could not remove existing destination File '%s', Exception: %s" % (outFile, inst)) #log error
            sys.exit(1)
        LOGGER.debug("Move %s -> %s" % (inFile, outFile))
        shutil.copy(inFile, outFile)
    else:
        createDirectory(backupDir)
        LOGGER.debug("Copy %s -> %s" % (inFile, outFile))
        shutil.copy(inFile, outFile)
    md5_1 = md5_for_file(inFile)
    md5_2 = md5_for_file(outFile)
    if md5_1 != md5_2:
        LOGGER.error("MD5Error in copying '%s' -> '%s', Exception: %s, %s" % (inFile, outFile, md5_1, md5_2)) #log error
        sys.exit(1)
    return inFile


def md5_for_file(filename, block_size=2 ** 20, accessType='b'):
    """creates a md5 hash of file

    :param filename: file to create md5 for
    :type filename: str
    :param block_size: md5 blocksize, defaults to 2**20
    :type block_size: int, optional
    :param accessType: access type for file , defaults to 'b'
    :type accessType: str, optional
    :return: md5 hash
    :rtype: hexvalue
    """
    f = open(filename, 'r' + accessType)
    md5 = hashlib.md5()
    while True:
        data = f.read(block_size)
        if not data:
            break
        md5.update(data)
    md5Out = md5.hexdigest()
    f.close()
    return md5Out

def sha512_for_file(filename, block_size=2 ** 20, accessType='b'):
    """creates a sha512 hash of file

    :param filename: file to create md5 for
    :type filename: str
    :param block_size: md5 blocksize, defaults to 2**20
    :type block_size: int, optional
    :param accessType: access type for file , defaults to 'b'
    :type accessType: str, optional
    :return: sha512 hash 
    :rtype: hexvalue
    """
    f = open(filename, 'r' + accessType)
    md5 = hashlib.sha512()
    while True:
        data = f.read(block_size)
        if not data:
            break
        md5.update(data)
    md5Out = md5.hexdigest()
    f.close()
    return md5Out

def createBatchFile(filename, fileList, command):
    """creates a new batch file, with the format: command + "file (from filelist)"

    :param filename: name of the batch
    :type filename: string
    :param fileList: iterable of objects to appen after command, all none string object are written in quotes
    :type fileList: iterable
    :param command: command that should be performed in batch on elements from fileList
    :type command: string
    """
    backupFile(filename)
    file = open(filename, "w")
    for f in fileList:
        if isinstance(f, str):
            file.write('%s "%s" \n' % (command, f))
        else:
            file.write('%s ' % command)
            for x in f:
                file.write('"%s" ' % (x))
            file.write('\n')
    file.close()
    LOGGER.debug("Created batch file %s" % filename)


def createStandardLogger(loggerName, loggerFileName, newFile=False):
    """creates a new logger

    :param loggerName: name of the logger
    :type loggerName: str
    :param loggerFileName: path of the logfile
    :type loggerFileName: str
    :param newFile: create new logfile, defaults to False
    :type newFile: bool, optional
    :return: logger
    :rtype: logger
    """
    if newFile and os.path.exists(loggerFileName):
        backupFile(loggerFileName, "_%s.bck" % str(datetime.date.today()))
        os.remove(loggerFileName)
    outputDir = os.path.split(loggerFileName)[0]
    createDirectory(outputDir)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(loggerFileName, 'w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    # create formatter and add it to the handlers
    formatterFile = logging.Formatter('%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s:%(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatterFile)
    # add the handlers to logger
    logging.getLogger('').addHandler(ch)
    logging.getLogger('').addHandler(fh)
    logger = logging.getLogger(loggerName)
    return logger

def get_immediate_subdirectories(dir):
    """get the names of immediate subdirectories

    :param dir: directorie to search subdirectories in
    :type dir: str
    :return: list of complete subdirectories paths
    :rtype: list of str
    """
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def get_immediate_subdirectoriesLists(dirList, mask=None):
    """get the names of immediate subdirectories filtered by a file mask

    :param dirList: list of directories to search subdirectories in
    :type dirList: [str]
    :param mask: filemask see https://winscp.net/eng/docs/file_mask, defaults to None
    :type mask: str, optional
    :return: list of subdirectories
    :rtype: list of str
    """
    outList = []
    for dir in dirList:
        wholeList = [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]
        if mask is not None:
            wholeList = fnmatch.filter(wholeList, mask)
        outList.extend([os.path.join(dir,x) for x in wholeList])
    return outList

def read4DFile(inFile, startLine=None, headerLineStartsWith='Subject No.'):
    """[summary]

    :param inFile: file to read
    :type inFile: str
    :param startLine: startline to read from (inclusiv), defaults to None
    :type startLine: str, optional
    :param headerLineStartsWith: headerline of 4d file, defaults to 'Subject No.'
    :type headerLineStartsWith: str, optional
    :return: [[content of one line split by "/t"]]
    :rtype: list of list of str
    """
    if not os.path.exists(inFile):
        LOGGER.debug("File not existing: %s" % inFile)
        return None
    try: 
        text_file = open(inFile, "r")
        LOGGER.debug("Read 4D file %s into list" % inFile)
    except IOError as e:
        LOGGER.Error("Cannot load File: %s, Exception: %s" % (inFile, e))
    lines = text_file.readlines()
    text_file.close()
    if startLine is None:
        for ind, line in enumerate(lines):
            if line[:len(headerLineStartsWith)] == headerLineStartsWith:
                startLine = ind + 1
                break
    if startLine is None:
        LOGGER.Error("Didn't find data start in 4D file %s (startLine: %d, headerStarW: '%s')" % (inFile, startLine, headerLineStartsWith))
        return None
    data = [line.split("\t") for line in lines[startLine:]]
    return data

class QueueHandler(logging.Handler):
    """
    This is a logging handler which sends events to a multiprocessing queue.
    The plan is to add it to Python 3.2, but this can be copy pasted into
    user code for use with earlier Python versions.
    """
    def __init__(self, queue):
        """Initialise an instance, using the passed queue.

        :param queue: queue
        :type queue: list
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def emit(self, record):
        """Emit a record. Writes the LogRecord to the queue.

        :param record: record to emit
        :type record: record
        """
        try:
            ei = record.exc_info
            if ei:
                dummy = self.format(record) # just to get traceback text into record.exc_text
                record.exc_info = None  # not needed any more
            self.queue.put_nowait(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def listener_configurer(loggerName, loggerFileName, newFile):
    """Because you'll want to define the logging configurations for listener and 
    workers, the listener and worker process functions take a configurer parameter which is a callablefor configuring logging for that process.
    These functions are also passed the queue, which they use for communication.
    In practice, you can configure the listener however you want, but note that in this
    simple example, the listener does not apply level or filter logic to received
    records.
    In practice, you would probably want to do ths logic in the worker processes, to avoid
    sending events which would be filtered out between processes.
    The size of the rotated files is made small so you can see the results easily.

    :param loggerName: name of the logger
    :type loggerName: str
    :param loggerFileName: name of logfile
    :type loggerFileName: str
    :param newFile: create a new logfile
    :type newFile: bool
    """
    if newFile and os.path.exists(loggerFileName):
        backupFile(loggerFileName, "_%s.bck" % str(datetime.date.today()))
        try:
            os.remove(loggerFileName)
        except Exception as inst:
            LOGGER.error("Could not remove existing destination File '%s', Exception: %s" % (loggerFileName, inst)) #log error
            sys.exit(1)
    outputDir = os.path.split(loggerFileName)[0]
    createDirectory(outputDir)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(loggerFileName, 'w')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    # create formatter and add it to the handlers
    formatterFile = logging.Formatter('%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s:%(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatterFile)
    # add the handlers to logger
    logging.getLogger('').addHandler(ch)
    logging.getLogger('').addHandler(fh)
    
def listener_process(queue, configurer, loggerName, loggerFileName, newFile=False):
    """This is the listener process top-level loop: wait for logging events
    (LogRecords)on the queue and handle them, quit when you get a None for a LogRecord.

    :param queue: to wait for logging events
    :type queue: list
    :param configurer: configurer to be configured
    :type configurer: configurer
    :param loggerName: name of the logger
    :type loggerName: str
    :param loggerFileName: name of the loggin file
    :type loggerFileName: str
    :param newFile: create a new file, defaults to False
    :type newFile: bool, optional
    """
    configurer(loggerName, loggerFileName, newFile)
    while True:
        try:
            record = queue.get()
            if record is None: # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record) # No level or filter logic applied - just do it!
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            import sys
            import traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def parseParameterDefinition(value, paraName):
    """if the parameter name has the format xy_int_xy or xy__float_xy returns the value as int/float else the original value

    :param value: value of the parameter
    :type value: object
    :param paraName: parameter name
    :type paraName: str
    :return: value as int/float/original value
    :rtype: object
    """
    name = paraName.split('_')[0]
    typ = paraName.split('_')[1]
    if typ == "int":
        return int(value)
    elif typ == "float":
        return float(value)
    else:
        return value

def parse_parameter_definition(value, paraName): 
    """if the parameter name has the format xy_int_xy or xy__float_xy returns the value as int/float else the original value

    :param value: value of the parameter
    :type value: object
    :param paraName: parameter name
    :type paraName: str
    :return: value as int/float/original value
    :rtype: object
    """
    name = paraName.split('_')[0]
    typ = paraName.split('_')[1]
    if typ == "int":
        return int(value)
    elif typ == "float":
        return float(value)
    else:
        return value

def parse_parameter_definition_traits(value, typ):
    """if the type is "int" or "float" returns the value as int or float, else return original value

    :param value: value of the type
    :type value: object
    :param typ: type as string
    :type typ: str
    :return: value as int/float/original value
    :rtype: object
    """
    if typ == "int":
        return int(value)
    elif typ == "float":
        return float(value)
    else:
        return value


def getDataFromFile_traits(filename, parseMask, parseParameters):
    """gets traits from file

    :param filename: path of the file
    :type filename: str
    :param parseMask: regular expression pattern must have same amount of groups as parseParameters have elements
    :type parseMask: str
    :param parseParameters: list of parameters to parse
    :type parseParameters: list of parameters
    :return: list of the parsed parameters
    :rtype: list
    """
    fileformat = re.compile(parseMask)
    m = fileformat.search(os.path.split(filename)[-1])
    if m is None:
        return None
    else:
        outList = []
        for para in range(len(parseParameters)):
            outList.append(parse_parameter_definition_traits(m.group(para+1), parseParameters[para].type))
        outList.append(filename)
    return outList

def getDataFromFile(filename, parseMask, parseParameters):
    """returns data parsed from filename (without path)

    :param filename: file name (base)
    :type filename: string
    :param parseMask: regular expression pattern
    :type parseMask: str
    :param parseParameters: parameters to parse
    :type parseParameters: list
    :raises Exception: different values
    :return: [subj, area, day, type(Lesion or not)]
    :rtype: list
    """
    if str(type(parseParameters)).find("traits.") >= 0:
       return getDataFromFile_traits(filename, parseMask, parseParameters)
    elif isinstance(parseMask, (list, tuple)):#parse list of filesmasks
        filename_local = filename
        out_dict = {}
        for idx, parseMask_loc in enumerate(parseMask[::-1]):
            filename_local, fname = os.path.split(filename_local)
            if parseMask_loc is not None:
                ol = getDataFromFile(fname, parseMask_loc, parseParameters)
                if ol is None:
                    return None
                for value, param in zip(ol[:-1], parseParameters):
                    if value is None:
                        continue
                    if param in out_dict:
                        if out_dict[param] != value:
                            raise Exception
                    out_dict[param] = value
        outList = []
        for param in parseParameters:
            if param in out_dict:
                outList.append(out_dict[param])
            else:
                outList.append(None)
        outList.append(filename)
        return outList
    else:
        fileformat = re.compile(parseMask)
        m = fileformat.search(os.path.split(filename)[-1])
        if m is None:
            return None
        else:
            outList = [parse_parameter_definition(m.group(para), para) for para in parseParameters]
            outList.append(filename)
            return outList

class NumPyArangeEncoder(json.JSONEncoder):
    """Class for encoding NumpyArray to json (helper class for json encoding)

    :param json: json to encode
    :type json: json
    """
    def default(self, obj):
        """returns ndarrays as list or encodes other object to json

        :param obj: to encode
        :type obj: object
        :return: list or json str
        :rtype: list, json str
        """
        from numpy import ndarray 
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def read_ptp_file(filename):
    """loads json files and returning data as dict

    :param filename: path of file
    :type filename: str
    :return: dict containing the loaded data, None when exception occurred
    :rtype: dict
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                points = json.load(f)
        except Exception as inst:
            LOGGER.error("Error reading pointfile '%s', Exception: %s"%(filename, inst))
            return None
    else:
        LOGGER.info("Not existing pointfile '%s'" % filename)
        return None
    return points

def write_ptp_file(filename, points):
    """creates a new file and stores points in json format

    :param filename: path of the new file
    :type filename: str
    :param points: points to store
    :type points: object
    :return: points at success or none at failure
    :rtype: object
    """
    backupFile(filename)
    try:
        f = open(filename, 'w') 
        if f:
            json.dump(points, f, sort_keys=False, indent=4)
        f.close()
    except Exception as inst:
        LOGGER.error("Error writing pointfile '%s', Exception: %s" % (filename, inst))
        f.close()
        os.remove(filename)
        return None
    return points

def read_ptx_file(filename):
    """reads points from json file with jsonpickle

    :param filename: path of file to read
    :type filename: str
    :return: points in dict format or none at failure
    :rtype: dict
    """
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                json_str = f.read()
                points = jsonpickle.decode(json_str)
        except Exception as inst:
            LOGGER.error("Error reading pointfile '%s', Exception: %s" % (filename, inst))
            return None
    else:
        LOGGER.info("Not existing pointfile '%s'" % filename)
        return None
    return points

def merge_dicts(*dict_args):
    """Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.

    :return: merged dict
    :rtype: dict
    """

    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def convert_string2ptx(input):
    """decode a json to dict format

    :param input: string to convert
    :type input: str
    :return: dict
    :rtype: dict
    """
    return jsonpickle.decode(input)

def convert_ptx2string(input):
    """Use jsonpickle to transform the object into a JSON string.

    :param input: object to transform
    :type input: object
    :return: object in JSON string format
    :rtype: JSON string
    """
    return jsonpickle.encode(input)

def write_ptx_file(filename, points):
    """creates file storing points in json format with jsonpickle

    :param filename: path of file to create
    :type filename: str
    :param points: points to store
    :type points: object
    :return: points at success, None at failure
    :rtype: object
    """
    backupFile(filename)
    try:
        f = open(filename, 'w') 
        if f:
            pickled = jsonpickle.encode(points)
            f.write(pickled)
        f.close()
    except Exception as inst:
        LOGGER.error("Error writing pointfile '%s', Exception: %s" % (filename, inst))
        f.close()
        os.remove(filename)
        return None
    return points

def cross_sets(args):
    """transforms list of list to one liste

    :param args: list of list
    :type args: [[object]]
    :return: [object]
    :rtype: list
    """
    ans = [[]]
    for arg in args:
        ans = [x + [y] for x in ans for y in arg]
    return ans

def cross_lists(args):
    """ transforms list of list to list of tupel like: [[1],[2],[3],[4],[5]] -> [((((1, 2), 3), 4), 5)]

    :param args: list of list to transform
    :type args: [[object]]
    :return: [()]
    :rtype: list of tupel
    """
    ans = list(itertools.product(args[0], args[1]))
    for ls in args[2:]:
        ans = list(itertools.product(ans,ls))
    return ans

def cross_it(*sets):
    """ returns crossproduct of sets

    :yield: next product
    :rtype: generator
    """
    wheels = list(map(iter, sets)) # wheels like in an odometer
    digits = [next(it) for it in wheels]
    while True:
        yield digits[:]
        for i in range(len(digits) - 1, -1, -1):
            try:
                digits[i] = next(wheels[i])
                break
            except StopIteration:
                wheels[i] = iter(sets[i])
                digits[i] = next(wheels[i])
        else:
            break


def dogroupby(k):
    """sorts k and remove duplicates

    :param k: input list
    :type k: list
    :return: new sorted list with unique elements
    :rtype: list
    """
    ks = sorted(k)
    return [i for i, _ in itertools.groupby(ks)]

def main_is_frozen():
    """check if main is frozen

    :return: if main is frozen
    :rtype: bool
    """
    return (hasattr(sys, "frozen") or # new py2exe
           hasattr(sys, "importers") # old py2exe
           or imp.is_frozen("__main__")) # tools/freeze

def combine_files_by_keys(study_data, keyIndexList):
    """combine files with keys

    :param study_data: data to combine
    :type study_data: iterable
    :param keyIndexList: key index list 
    :type keyIndexList: list of indexes
    :return: combined data
    :rtype: list
    """
    keyIndexList = list(keyIndexList)
    keyIndexList.sort()
    def keyFunction(key):
        return [key[idx] for idx in keyIndexList]
    studyDataCorrList = dogroupby([[zz[key] for key in list(keyIndexList)] for zz in study_data]) # list of unique
    combinedDataList = [[key, [z for z in [x for x in study_data if keyFunction(x) == key]]] for key in studyDataCorrList]
    return combinedDataList

def check_study_input_data(filenames, INPUT_FILE_PARSE_MASK, INPUT_FILE_PARSE_PARAMETERS, NumberOfCorrespondingFiles, wait_input=False, do_test=True, key_remove_list=[], return_combined=False):
    """prepares study data 

    :param filenames: filenames to get
    :type filenames: str
    :param INPUT_FILE_PARSE_MASK: parse mask
    :type INPUT_FILE_PARSE_MASK: str
    :param INPUT_FILE_PARSE_PARAMETERS: parameters to parse
    :type INPUT_FILE_PARSE_PARAMETERS: list
    :param NumberOfCorrespondingFiles: number of correspondending files
    :type NumberOfCorrespondingFiles: int
    :param wait_input: wait on user input, defaults to False
    :type wait_input: bool, optional
    :param do_test: test if data is complete or just parse data, defaults to True
    :type do_test: bool, optional
    :param key_remove_list: elements to ignore in keyindexlist, defaults to []
    :type key_remove_list: list, optional
    :param return_combined: return combined data or single values, defaults to False
    :type return_combined: bool, optional
    :raises Exception: no files with correct name found
    :return: study data, errors , combined study data(when return combined is set)
    :rtype: list , list, list
    """
    study_data = [getDataFromFile(x, INPUT_FILE_PARSE_MASK, INPUT_FILE_PARSE_PARAMETERS) for x in filenames]
    study_data_wrong_name = []
    if None in study_data:
        study_data_wrong_name = [filenames[idx] for idx,data in enumerate(study_data) if data is None]
    study_data = list(map(tuple, [x for x in study_data if x is not None]))
    if not do_test:
        if return_combined:
            return study_data, None ,None
        return study_data, None
    if len(study_data) < 1:
        raise Exception('No files with correct name')
    duplicates = []
    if len(study_data) != len(set(study_data)):
        import collections.abc
        counter = collections.Counter(study_data)
        duplicates = [x for x in list(counter.items()) if x[1] > 1]
        print(counter)
    keyIndexList = set(range(len(study_data[0]) - 1)) - set(key_remove_list)
    combinedDataList = combine_files_by_keys(study_data, keyIndexList) # list of unique
    studyDataUniqueList = [list(set([x[index] for x in study_data])) for index in range(len(study_data[0]) - 1)]
    studyDataKeyList = [studyDataUniqueList[key] for key in keyIndexList]
    allDataList = cross_sets(studyDataKeyList)
    foundDataList = [[zz[key] for key in keyIndexList] for zz in study_data]
    testList1 = set([tuple(x) for x in allDataList])
    testList2 = set([tuple(x) for x in foundDataList])
    missingData = (testList1 - testList2)
    combinedDataList_missing = []
    combinedDataList_duplicates = []
    if NumberOfCorrespondingFiles > 0:
        combinedDataList_good = [x for x in [x for x in combinedDataList if len(x[-1]) == NumberOfCorrespondingFiles]]
        combinedDataList_missing = [x[-1] for x in [x for x in combinedDataList if len(x[-1]) < NumberOfCorrespondingFiles]]
        combinedDataList_duplicates = [x[-1] for x in [x for x in combinedDataList if len(x[-1]) > NumberOfCorrespondingFiles]]

    err_data = {}
    if missingData or combinedDataList_missing or study_data_wrong_name or duplicates or combinedDataList_duplicates:
        if study_data_wrong_name:
            study_data_wrong_name.sort()
            LOGGER.warning("Wrong named files:\n%s" % "\n".join([str(x) for x in study_data_wrong_name]))
            print("Wrong named files:\n%s" % "\n".join([str(x) for x in study_data_wrong_name]))
            err_data['wrong_name'] = study_data_wrong_name
        if duplicates:
            duplicates.sort()
            LOGGER.warning("Duplicate files:\n%s" % "\n".join([str(x) for x in duplicates]))
            print("Duplicate files:\n%s" % "\n".join([str(x) for x in duplicates]))
            err_data['duplicates'] = duplicates
        if combinedDataList_duplicates:
            combinedDataList_duplicates.sort()
            LOGGER.warning("Duplicate files after combining:\n%s" % "\n".join([str(x) for x in combinedDataList_duplicates]))
            print("Duplicate files after combining:\n%s" % "\n".join([str(x) for x in combinedDataList_duplicates]))
            err_data['combinedDataList_duplicates'] = combinedDataList_duplicates
        if missingData:
            missingData = list(missingData)
            missingData.sort()
            LOGGER.warning("Missing Data:\n%s" % "\n".join([str(x) for x in missingData]))
            print("Missing Data:\n%s" % "\n".join([str(x) for x in missingData]))
            err_data['missingData'] = missingData
        if combinedDataList_missing:
            combinedDataList_missing = list(combinedDataList_missing)
            combinedDataList_missing.sort()
            LOGGER.warning("Missing Data:\n%s" % "\n".join([str(x) for x in combinedDataList_missing]))
            print("Missing Data:\n%s" % "\n".join([str(x) for x in combinedDataList_missing]))
            err_data['combinedDataList_missing'] = combinedDataList_missing
        if wait_input:
            input("Press Enter to continue or ^C to quit...")
    if return_combined:
        return study_data, err_data, combinedDataList
    else:
        return study_data, err_data

def check_study_data(basepath, file_pattern, INPUT_FILE_PARSE_MASK, INPUT_FILE_PARSE_PARAMETERS, NumberOfCorrespondingFiles, search_depth=1, wait_input=False, do_test=True, key_remove_list=[]):
    """gets study data

    :param basepath: base path of data
    :type basepath: str
    :param file_pattern: pattern of the files
    :type file_pattern: str
    :param INPUT_FILE_PARSE_MASK: parse mask
    :type INPUT_FILE_PARSE_MASK: str
    :param INPUT_FILE_PARSE_PARAMETERS: parameters to get
    :type INPUT_FILE_PARSE_PARAMETERS: list
    :param NumberOfCorrespondingFiles: number of corresponding files
    :type NumberOfCorrespondingFiles: int
    :param search_depth: search depth, defaults to 1
    :type search_depth: int, optional
    :param wait_input: wait for user input, defaults to False
    :type wait_input: bool, optional
    :param do_test: check data, defaults to True
    :type do_test: bool, optional
    :param key_remove_list: keys to ignore in key index list, defaults to []
    :type key_remove_list: list, optional
    :return: returns only study data
    :rtype: list
    """
    filenames = getAllFiles(basepath, file_pattern, search_depth)
    return check_study_input_data(filenames, INPUT_FILE_PARSE_MASK, INPUT_FILE_PARSE_PARAMETERS, NumberOfCorrespondingFiles, wait_input=wait_input, do_test=do_test, key_remove_list=key_remove_list)[0]

def check_study_data_with_err_data(basepath, file_pattern, INPUT_FILE_PARSE_MASK, INPUT_FILE_PARSE_PARAMETERS, NumberOfCorrespondingFiles, search_depth=1, wait_input=False, do_test=True, key_remove_list=[]):
    """gets study data with errors

    :param basepath: base path of data
    :type basepath: str
    :param file_pattern: pattern of the files
    :type file_pattern: str
    :param INPUT_FILE_PARSE_MASK: parse mask
    :type INPUT_FILE_PARSE_MASK: str
    :param INPUT_FILE_PARSE_PARAMETERS: parameters to get
    :type INPUT_FILE_PARSE_PARAMETERS: list
    :param NumberOfCorrespondingFiles: number of corresponding files
    :type NumberOfCorrespondingFiles: int
    :param search_depth: search depth, defaults to 1
    :type search_depth: int, optional
    :param wait_input: wait for user input, defaults to False
    :type wait_input: bool, optional
    :param do_test: check data, defaults to True
    :type do_test: bool, optional
    :param key_remove_list: keys to ignore in key index list, defaults to []
    :type key_remove_list: list, optional
    :return: returns study data and errors
    :rtype: list, list
    """
    filenames = getAllFiles(basepath, file_pattern, search_depth)
    return check_study_input_data(filenames, INPUT_FILE_PARSE_MASK, INPUT_FILE_PARSE_PARAMETERS, NumberOfCorrespondingFiles, wait_input=wait_input, do_test=do_test, key_remove_list=key_remove_list)

def writeData(data, filename, method=lambda data,f: pickle.dump(data,f,-1)):
    """write to a already existing file using pickle

    :param data to write
    :type data: object
    :param filename: path of the file to write into
    :type filename: str
    :param f: function to use for writing
    :type f: pickle.dump
    :param method: method of writing, defaults to lambda data
    :type method: method, optional
    """
    try:
        with open(filename, 'wb') as f:
            method(data,f)
    except Exception as inst:
        LOGGER.error("Could not marshale File '%s', Exception: %s" % (filename, inst)) #log error
        raise

def readData(filename, method=lambda x: pickle.load(x)):
    """reads a pickle file

    :param filename: path of file to read from
    :type filename: str
    :param method: method for reacding, defaults to lambdax:pickle.load(x)
    :type method: method, optional
    :return: read data
    :rtype: object
    """
    try:
        LOGGER.debug("Try read marshaled File '%s'" % (filename)) #log error
        with open(filename, 'rb') as f:
            data = method(f)
        return data
    except Exception as inst:
        LOGGER.debug("Could not read marshaled File '%s', Exception: %s" % (filename, inst)) #log error
        raise

def writeDataJSON(data, filename, method=lambda data,f: pickle.dump(data,f,-1)): 
    """writes data in json format

    :param data: to write
    :type data: object
    :param filename: file to write into
    :type filename: str
    :param f: function to write with
    :type f: pickle.dump
    :param method: method , defaults to lambdadata
    :type method: function, optional
    """
    writeData(data, filename, method=lambda data,f: f.write(jsonpickle.encode(data)))

def readDataJSON(filename):
    """reads json file

    :param filename: file to read
    :type filename: str
    :return: read data
    :rtype: object
    """
    return readData(filename, method=lambda x: jsonpickle.decode(x.read()))

def multiple_replacer(replace_dict):
    """creates lambda function which replaces strings, replacement are defined by the dict

    :param replace_dict: dict which defines replacement rules of the lambda function
    :type replace_dict: dict
    :return: lambda function which replaces strings
    :rtype: lambda function    
    """
    key_values = list(replace_dict.items())
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values]), re.M)
    return lambda string: pattern.sub(replacement_function, string)

def multiple_replace(string, replace_dict):
    """replaces the substrings

    :param string: in which substrings should be replaced
    :type string: str
    :param replace_dict: dict that defines replacements
    :type replace_dict: dict
    :return: string with replaced substrings
    :rtype: str
    """
    return multiple_replacer(replace_dict)(string)

def export_for_stat_format(data_dictionary, output_file):
    """writes data to file

    :param data_dictionary: dictionary containing data, must contain 'data' keyword 
    :type data_dictionary: dict
    :param output_file: file to write in
    :type output_file: str
    """
    backupFile(output_file)
    header = data_dictionary['header']
    with open(output_file, "w") as outfile:
        outfile.write("\t".join(header) + "\n")
        for dt in data_dictionary['data']:
            outfile.write("\t".join(dt) + "\n")

