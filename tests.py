import scipy.io.wavfile as wf
import plotter
import main


def run_tests():


    ### Test random sampling ###

    print(main.latent_vector_uniform(50))
    print(main.latent_vector_normal(50))

    ### Test plot_wav function ###
    plotter.plot_wav('./testwav.wav')

    plotter.plot_wav('./testwav.wav', size=100)

    ### Test what kind of data is returned by scipys wavfile.read() ###
    a, b = wf.read('./testwav.wav')
    print(a)
    print("####")
    print(b)

if __name__ == '__main__':
    run_tests()