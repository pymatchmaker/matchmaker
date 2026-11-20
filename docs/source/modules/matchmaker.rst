matchmaker
===========

.. autoclass:: matchmaker.Matchmaker
   :members: run
   :show-inheritance:
   :special-members: __init__

Every score follower implements the ``OnlineAlignment`` interface.

.. autoclass:: matchmaker.base.OnlineAlignment
   :members: step, run, get_current_position
   :special-members: __call__

.. autoclass:: matchmaker.external.OnlineParangonarAlignment
   :show-inheritance:
   :special-members: __init__
