import os


class BasicProcess(object):
    """
    Process block is a basic running module for this repo, it will run the process by checking if function has been
    ran before, or be forced to re-run the process again
    """
    def __init__(self, name, path, func=None):
        """
        :param name:name of the process, this will be used for the state file name
        :param path: path to where the state file will be stored
        :param func: process function, if None then it will be child class's process() function
        """
        self.name = name
        self.path = path
        if func is None:
            self.func = self.process
        else:
            self.func = func
        self.state_file = os.path.join(self.path, '{}_state.txt'.format(self.name))

    def process(self, **kwargs):
        raise NotImplementedError()

    def run(self, force_run=False, **kwargs):
        """
        Run the process
        :param force_run: if True, then the process will run no matter it has completed before
        :param kwargs:
        :return:
        """
        # check if state file exists
        state_exist = os.path.exists(self.state_file)
        # run the function if force run or haven't run before
        if force_run == 1 or state_exist == 0:
            print(('Start running {}'.format(self.name)))
            # write state log as incomplete
            with open(self.state_file, 'w') as f:
                f.write('Incomplete\n')

            # run the process
            self.func(**kwargs)

            # write state log as complete
            with open(self.state_file, 'w') as f:
                f.write('Finished\n')
        else:
            # if haven't run before, run the process
            if not self.check_finish():
                self.func(**kwargs)
        return self

    def check_finish(self):
        """
        check if state file exists
        :return: True if it has finished
        """
        state_exist = os.path.exists(self.state_file)
        if state_exist:
            with open(self.state_file, 'r') as f:
                a = f.readlines()
                if a[0].strip() == 'Finished':
                    return True
        return False
