# save gif
import imageio
import os
ims = []
LEARNING_STEPS = 50
for i in range(LEARNING_STEPS):
    fname = 'generated_plot_%d.png' % i
    dir = 'bagan_gp_results/'
    if fname in os.listdir(dir):
        print('loading png...', i)
        im = imageio.imread(dir + fname, 'png')
        ims.append(im)
print('saving as gif...')
imageio.mimsave(dir + 'training_demo.gif', ims, fps=3)