# $Id: StandardLogger.py 12592 2021-08-23 07:23:24Z ndrews $
"""
###############################################################################
standard_logger.py
###############################################################################

Purpose
===============================================================================

Create standard logging facility:

    1. logging to file
    2. logging to commandline


Usage
===============================================================================

Should be included in all python files as follows

In modules/libraries/API (already done when using python_module_template in VS)::

    import logging

    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Importing %s, version %s" % (__name__, __svnData__()['id'])

In scripts (already done when using python_script_template in VS)::

    import standard_logger

    OUTPUTPATH = os.getcwd()
    LOGGER_NAME = os.path.splitext(os.path.split(inspect.getfile(
    inspect.currentframe()))[1])[0]
    LOGGER = standard_logger.create_standard_logger(LOGGER_NAME, os.path.join(OUTPUTPATH,
        LOGGER_NAME + ".log"), create_newfile = True)
    LOGGER.setLevel(logging.INFO) #can be used to change level of detail 
                                  #messages in commandline
    LOGGER.info("SVNId: %s" % __svnData__()['id'])

"""

__version__ = "$Revision: 12592 $"


def __svnData__():
    """
    | $Author: ndrews $
    | $Date: 2021-08-23 09:23:24 +0200 (Mo., 23 Aug 2021) $
    | $Rev: 12592 $
    | $URL: http://sw-server:8090/svn/ImageProcessingLibrary/Python/proDERM_ImageAnalysisLibrary/ImageAnalysis/StandardLogger.py $
    | $Id: StandardLogger.py 12592 2021-08-23 07:23:24Z ndrews $
    """
    # only for documentation purpose
    return {
            'author': "$Author: ndrews $".replace('$', '').replace('Author:', '').strip(),
            'date': "$Date: 2021-08-23 09:23:24 +0200 (Mo., 23 Aug 2021) $".replace('$', '').replace('Date:', '').strip(),
            'rev': "$Rev: 12592 $".replace('$', '').replace('Rev:', '').strip(),
            'id': "$Id: StandardLogger.py 12592 2021-08-23 07:23:24Z ndrews $".replace('$', '').replace('Id:', '').strip()
            }

import logging
import logging.handlers
import shutil
import sys
import os
import os.path
import datetime


def __backup_file(inFile, backupExtension=".bck", maxBackup=-1):
    """creates backupfile, older files are saved with timestamp
    :param inFile: filename of file to backup
    :type inFile: string(filename)
    :param backupExtension: extension added to backupfiles
    :type backupExtension: string(".bck")
    :param maxBackup: number of kept backupfiles, -1 is all
    :type maxBackup: type(-1)
    :return:  filename of backup file
    :rtype:
    """
    LOGGER = logging.getLogger(__name__)
    if not(os.path.exists(inFile)):
        LOGGER.info("No Backup possible, %s does not exist" % (inFile))
        return
    backupFileName = inFile + backupExtension
    if maxBackup != 0:
        ct = 0
        while os.path.exists(backupFileName) and ((maxBackup < 0) or (ct < maxBackup)):
            backupFileName = inFile + ".%d" % ct + backupExtension
            ct += 1
    LOGGER.info("copy %s -> %s" % (inFile, backupFileName))
    try:
        shutil.copy(inFile, backupFileName)
    except Exception as inst:
        LOGGER.error("Could copy '%s -> %s', Exception: %s" % (inFile, backupFileName, inst))
        sys.exit(1)
    return backupFileName


def __create_directory(dirname):
    """creates directory structure if not existing
    .. warning:: Quits program if directory could not be created
    :param dirname: 
    :type dirname: string
    :return: True if directory was created/existed, exits program if
    :rtype: Bool
    """
    LOGGER = logging.getLogger(__name__)
    if not os.path.exists(dirname):
        LOGGER.info("Create path %s" % dirname)
        try:
            os.makedirs(dirname)
        except Exception as inst:
            LOGGER.critical("Could nor create directory '%s', Exception: %s" % (dirname, inst))
            sys.exit(1)
            return False
    LOGGER.info("Path %s exist" % dirname)
    return True


def create_standard_logger(logger_name, logger_filename, create_newfile=False):
    """creates standard logger for python
    file logger:
        includes all (DEBUG) messages
    command-line logger:
        imcludes WARN, ERROR, CRITICAL messages
    .. warning:: not multithreaded!
    :param logger_name: name of logger (typically script name)
    :type logger_name: string
    :param logger_filename: filename of log-file
    :type logger_filename: string
    :param create_newfile: if True, create always a new log file, older are backed up
    :type create_newfile: bool(False)
    :return: logger with 
    :rtype: logger object
    """
    try:
        if os.path.exists(logger_filename):
            os.rename(logger_filename, logger_filename + "test.log")
            os.rename(logger_filename + "test.log", logger_filename)
    except:
        logger_filename_name, logger_filename_ext = os.path.splitext(logger_filename)
        logger_filename = logger_filename_name + "_%d" % os.getpid() + logger_filename_ext
    if create_newfile and os.path.exists(logger_filename):
        __backup_file(logger_filename, "_%s.bck" % str(datetime.date.today()))
        os.remove(logger_filename)
    outputdir = os.path.split(logger_filename)[0]
    __create_directory(outputdir)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logger_filename, 'w')
    #logfile = os.path.abspath("C:\\Experimente\\mylogfile.log")
    #fh = ConcurrentRotatingFileHandler(logger_filename, 'w', 0, 0)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter_file = logging.Formatter('%(asctime)s - %(name)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s:%(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter_file)
    # add the handlers to logger
    logging.getLogger('').addHandler(ch)
    logging.getLogger('').addHandler(fh)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.info("Importing Module standardLogger")
    logger.info("SVNID: %s" % __svnData__()['id'])
    return logger



def create_network_logger(logger_name, host='localhost', debug=False):
    """creates standard logger for python
    file logger:
        includes all (DEBUG) messages
    command-line logger:
        imcludes WARN, ERROR, CRITICAL messages
    .. warning:: not multithreaded!
    :param logger_name: name of logger (typically script name)
    :type logger_name: string
    :param logger_filename: filename of log-file
    :type logger_filename: string
    :param create_newfile: if True, create always a new log file, older are backed up
    :type create_newfile: bool(False)
    :return: logger with 
    :rtype: logger object
    """
    #try:
    #    os.system('start /min python.exe network_logger.py')
    #except:
    #    pass
    socketHandler = logging.handlers.SocketHandler(host,
                        logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    ch = logging.StreamHandler()
    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s:%(funcName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logging.getLogger('').addHandler(socketHandler)
    logging.getLogger('').addHandler(ch)
    logger = logging.getLogger(logger_name)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    #logger.setLevel(logging.INFO)
    logger.debug("Importing Module standardLogger")
    logger.debug("SVNID: %s" % __svnData__()['id'])
    return logger


def test_network_logger_running():
    dos_command = 'tasklist /fi "imagename eq network_logger.exe"'# | find /i "network" /c'
    import subprocess
    out = subprocess.check_output(dos_command, stderr=subprocess.STDOUT, shell=True)
    if 'network' not in out:
        pass
    print(out)