import numpy as np

set_hand = np.random.randint(1,4)

if (set_hand == 1):
    print('Rock')
    print('# # # # #')
    print('# . . . #')
    print('# . . . #')
    print('# . . . #')
    print('# # # # #')
elif (set_hand == 2):
    print('paper')
    print('. . . . .')
    print('. # # # .')
    print('. # # # .')
    print('. # # # .')
    print('. . . . .')
elif (set_hand == 3):
    print('scissors')
    print('# # . . #')
    print('# # . # .')
    print('. . # . .')
    print('# # . # .')
    print('# # . . #')
else:
    print('Random out of scope')