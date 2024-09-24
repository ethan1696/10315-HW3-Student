import io
import contextlib
import traceback
import concurrent.futures

def safe_clear_output():
    try:
        from IPython.display import clear_output
        clear_output()
    except ImportError:
        pass

class Test:
    def __init__(self, name, testFn, timeout_seconds=10):
        assert len(name) < 100
        self.name = name
        self.testFn = testFn
        self.timeout_seconds = timeout_seconds
    
    def run_test(self):
        stdout_buffer = io.StringIO()

        def wrapped_test_fn():
            with contextlib.redirect_stdout(stdout_buffer):
                self.testFn()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(wrapped_test_fn)
            try:
                future.result(timeout=self.timeout_seconds)  
                captured_stdout = stdout_buffer.getvalue()
                return True, captured_stdout, None, None  
            except concurrent.futures.TimeoutError:
                captured_stdout = stdout_buffer.getvalue()
                return False, captured_stdout, "Test timed out", None
            except Exception as e:
                captured_stdout = stdout_buffer.getvalue()
                error_name = type(e).__name__
                error_traceback = traceback.format_exc()
                return False, captured_stdout, error_name, error_traceback

class TestSuite:
    def __init__(self, name):
        self.name = name
        self.tests = []

    def add_test(self, test):
        self.tests.append(test)

    def run_tests(self, test_ids=None, show_details=True):
        max_len = 130
        def pad_out_title(title, passed, dash_char="-"):
            if len(title) % 2 != 0:
                title += " "
            num_dashes = (max_len - len(title) - 2) // 2
            title_str = (dash_char * num_dashes) + " " + title + " " + (dash_char * num_dashes)

            if passed == None:
                return title_str

            if passed:
                title_str = "\033[32m" + title_str + "\033[0m"
            else:
                title_str = "\033[31m" + title_str + "\033[0m"
            
            return title_str
        
        print(pad_out_title(self.name, None, dash_char="="))
            
        if test_ids == None:
            test_ids = [i for i in range(len(self.tests))]
        
        outputs = []
        results = []

        summary_str = ""

        for test_id in test_ids:
            passed, stdout, error_name, error_traceback = self.tests[test_id].run_test()
            
            result = "\033[32mPASSED\033[0m" if passed else "\033[31mFAILED\033[0m"
            extension = f"\n\033[31m    Failure reason: {error_name}\033[0m" if error_name != None else ""

            test_output = f"Test {test_id}: {result} - {self.tests[test_id].name}{extension}"
            summary_str += test_output + "\n"
            if show_details:
                print(test_output)
            if len(stdout) > 0 or error_traceback != None:
                outputs.append((test_id, passed, self.tests[test_id].name, stdout, error_traceback))
            results.append(passed)
        

        safe_clear_output()

        
        if show_details:
            for test_id, passed, name, stdout, error_traceback in outputs:
                title = f"Test {test_id}: {name} - OUTPUTS"
                print(pad_out_title(title, passed))

                print(stdout)

                if error_traceback != None:
                    print(error_traceback)
        
        print(pad_out_title(
            f"SUMMARY - {self.name}", 
            all(results), 
            dash_char="="
        ))
        print(summary_str)

            