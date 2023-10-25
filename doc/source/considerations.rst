#########################################
Caveats, Concerns, & Other Considerations
#########################################

Some of this was hastily thrown together. Before publishing a paper using this code, there are few items that should be reviewed: (this is not exhaustive)


************************
Creating Ray Collections
************************

The ray-rotation code is somewhat convoluted and confusing.
See ``misc/gasimage.pdf`` for an image that illustrates the relationship between the various quantities.
On this topic, it may be worth writing additional tests verifying the correctness of the ray-rotation code.

I'm fairly confident that this works, but I haven't tested things varying the reference ray's latitude (in the frame of the observer); so far I have just used sky_latitude = 0.0).


***************
Other thoughts:
***************

- write tests to verify the correctness of the ray intersection code.
  (I suspect it's accurate, but it may be worth checking by comparing results to using ``YTRay``)

- Currently, when we save velocity channels saved just correspond to radiative transfer at a single frequency.
  Make sure that's the right thing to do for final mock image...
  (As an output of the rt code, this is definitely correct, but it's not clear if the creation of )

- Make sure that the conversion to brightness Temperature is correct.
  The conversion currently uses the frequency in each velocity bin.
  Should we be using the rest-frame velocity.

****************************
Potentially useful features:
****************************

- It's also worth noting the cython version of the radiative transfer is not currently in-use.
  There seems to be some minor bugs in it.
  This should be easy to fix and could provide a concrete speedup

- revisit the ``rescale_length_factor`` variable.
  This was hacked on and it's not clear to me if it still works.
  
- It's probably useful to allow the user to introduce a bulk velocity to the entire simulation domain, with respect to the observer.
