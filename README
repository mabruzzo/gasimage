This module nominally constructs HI images of gas clouds by under the assumption
that the gas is optically thin

The ray-rotation code is somewhat confusing. See misc/gasimage.pdf for an image that illustrates the relationship between the various quantities.


Some of this was hastily thrown together. Before publishing a paper
using this code, there are few items that should be reviewed: (this is not exhaustive)

- check the units on the radiative transfer.

- write tests to verify the correctness of the ray intersection code

- Write additional tests verifying the correctness of the ray-rotation code:

    - I'm fairly confident that this works. (Although I'm not completely
      convinced that the plots of the rays are completely correct).
    - I haven't tested things varying the reference ray's latitude (in the
      frame of the observer); so far I have just used sky_latitude = 0.0).

- Currently velocity channels just correspond to radiative transfer at
  a single frequency. Make sure that's the right thing to do.

- It might be worth making it possible for user's to introduce a bulk velocity
  to the entire simulation domain, with respect to the observer.

- Make sure that the conversion to brightness Temperature is correct.


It's also worth noting the cython version of the radiative transfer is not currently in-use. There seems to be some minor bugs in it.