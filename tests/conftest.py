import os
import shutil
from tempfile import mkdtemp

import pytest


_INDATADIR_OPT = "--indata-dir"

def pytest_addoption(parser):
    """
    This is a hook understood by pytest for parsing additional command line
    options.
    """

    parser.addoption(
        "--save-answer-dir", action="store", default = None,
        help = "path to the output directory where answers will be written."
    )

    parser.addoption(
        "--ref-answer-dir", action = "store", default = None,
        help = "path to the directory where answers can be read from."
    )

    parser.addoption(
        "--indata-dir", action = "store", default = None,
        help = "path to the directory where input-data can be read from."
    )


def pytest_collection_modifyitems(config, items):
    """
    This is a hook recognized by pytest for modifying properties of collected
    tests.

    We use it to conditionally skip answer-tests or tests that require an input
    directory, based on the availability of command line arguments
    """

    # define markers that provide hypothetical reasons why a test may be skipped
    skip_answer = pytest.mark.skip(
        reason = "need --save-answer-dir or --ref-answer-dir options to run")
    skip_indata = pytest.mark.skip(reason = "need --indata-dir option to run")

    missing_opt = lambda opt: config.getoption(opt) is None
    config_skip_answer = (missing_opt("--save-answer-dir") and
                          missing_opt("--ref-answer-dir"))
    config_skip_indata = missing_opt("--indata-dir")

    for item in items:
        if config_skip_answer and ("answer_test_config" in item.fixturenames):
            item.add_marker(skip_answer)
        elif config_skip_indata and ("indata_dir" in item.fixturenames):
            item.add_marker(skip_indata)

class AnswerTestConfig:
    def __init__(self, save_answer_dir = None, ref_answer_dir = None):
        self.save_answer_dir = save_answer_dir
        self.ref_answer_dir = ref_answer_dir

@pytest.fixture
def answer_test_config(request):
    """
    By defining this function, any function defined in the test-suite that 
    accepts an argument called ``answer_test_config`` will receive the result
    of this function.
    """
    if hasattr(request, "param"):
        raise RuntimeError("something might have changed")

    # come up with a name for the test
    test_name = f"{request.module.__name__}-{request.function.__name__}"

    def _testpath_from_cmdopt(opt):
        base_path = mkdtemp() if opt is None else request.config.getoption(opt)
        if base_path is None:
            return None, None
        return base_path, f"{base_path}/{test_name}/"

    # get the directory where reference answers are read from
    _, ref_answer_dir = _testpath_from_cmdopt("--ref-answer-dir")
    if (ref_answer_dir is not None) and (not os.path.isdir(ref_answer_dir)):
        raise ValueError(f"ref answer dir, {ref_answer_dir}, doesn't exist")

    # get the directory where answers are saved to. If the ref-dir is given
    # without a save-dir, then we create a temporary dir.
    _, save_answer_dir = _testpath_from_cmdopt("--save-answer-dir")
    if (ref_answer_dir is not None) and (ref_answer_dir == save_answer_dir):
        raise RuntimeError("the --ref-answer-dir & --save-answer-dir options "
                           "can't specify the same directory")
    elif (ref_answer_dir is not None) and (save_answer_dir is None):
        cleanup_dir, save_answer_dir = _testpath_from_cmdopt(None)
    else:
        cleanup_dir = None

    # ensure that the testspecific output-subdirectory actually exists
    if save_answer_dir is not None:
        os.makedirs(save_answer_dir, exist_ok = True)

    yield AnswerTestConfig(save_answer_dir = save_answer_dir,
                           ref_answer_dir = ref_answer_dir)

    # the following is executed after the answer-test is finished
    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir)

@pytest.fixture
def indata_dir(request):
    """
    By defining this function, any function defined in the test-suite that 
    accepts an argument called ``indata_dir`` will receive the result of this 
    function.
    """
    path = request.config.getoption(_INDATADIR_OPT)
    if not os.path.isdir(path):
        raise ValueError(f"--indata-dir specifies {path}: it doesn't exist")
    return f'{path}/'
