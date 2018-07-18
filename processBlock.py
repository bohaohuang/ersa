import os


class BasicProcess(object):
    def __init__(self, name, path, func=None):
        self.name = name
        self.path = path
        if func is None:
            self.func = self.process
        else:
            self.func = func

    def process(self, **kwargs):
        raise NotImplementedError()

    def run(self, force_run=False, **kwargs):
        # check if state file exists
        state_file = os.path.join(self.path, '{}_state.txt'.format(self.name))
        state_exist = os.path.exists(state_file)
        # run the function if force run or haven't run before
        if force_run == 1 or state_exist == 0:
            print(('Start running {}'.format(self.name)))
            # write state log as incomplete
            with open(state_file, 'w') as f:
                f.write('Incomplete\n')

            # run the process
            self.func(**kwargs)

            # write state log as complete
            with open(state_file, 'w') as f:
                f.write('Finished\n')
        else:
            # if haven't run before, run the process
            with open(state_file, 'r') as f:
                a = f.readlines()
                if a[0].strip() != 'Finished':
                    self.func(**kwargs)
