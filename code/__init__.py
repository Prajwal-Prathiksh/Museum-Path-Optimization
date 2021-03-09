###########################################################################
# Imports
###########################################################################
# Standard library imports
import os


###########################################################################
# Code
###########################################################################
def print_hello_world(ext=None):
    ''' Simple function which prints HelloWorld, with an optional parameter to
        print an additional string message

        Parameters:
        ----------
        ext: string, default=None
            Prints an additional message. If no argument is passed, prints the 
            current working directoy
    '''
    if ext == None:
        ext = os.getcwd()

    print(f'Hello World! \n{ext}')


###########################################################################
# Main Code
###########################################################################
if __name__ == '__main__':
    print_hello_world()

