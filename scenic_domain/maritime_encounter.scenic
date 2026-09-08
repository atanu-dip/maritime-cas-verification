
"""Maritime encounter scenario space (Scenic 3).

Scenic's role here is what a scenario description language is actually for:
declaring an *abstract space* of encounters -- distributions over the encounter
parameters plus constraints on which combinations are admissible -- rather than
a fixed list of test cases. Each sampled scene fixes one concrete assignment of
those parameters, which the Python domain bridge then instantiates into initial
positions and headings via the inverse DCPA/TCPA solve.

Parameterising by (DCPA, TCPA, relative geometry) rather than by position is
what makes the sampled set a controlled test suite: encounter difficulty is an
independent variable, and every sample is a genuine encounter.
"""

param encounter = 'head_on'          # head_on | crossing | overtaking

# encounter difficulty 
param dcpa_m = Range(0, 1500)        # metres at closest point of approach
param tcpa_s = Range(600, 1800)      # seconds to closest point of approach
param passing_side = Uniform(-1, 1)

# vessel particulars 
param own_vessel = Uniform('container_feeder', 'handysize_bulker',
                           'coastal_tanker', 'vlcc')
param tgt_vessel = Uniform('container_feeder', 'handysize_bulker',
                           'coastal_tanker', 'vlcc')

param own_speed_kn = Range(8, 16)
param tgt_speed_kn = Range(8, 16)

# COLREGS Rule 18 categories and Rule 19 visibility 
param own_status = 'power_driven'
param tgt_status = Discrete({'power_driven': 0.80,
                             'fishing': 0.10,
                             'restricted_manoeuvrability': 0.06,
                             'not_under_command': 0.04})
param visibility = Discrete({'good': 0.75, 'moderate': 0.18, 'restricted': 0.07})

#  target heading, conditioned on encounter type 
param head_on_heading    = Range(176, 184)
param overtaking_heading = Range(-12, 12)
param crossing_heading_s = Range(200, 340)   # target approaching from starboard
param crossing_heading_p = Range(20, 160)    # target approaching from port
param crossing_from_starboard = Discrete({1: 0.5, 0: 0.5})

# The Rule 13 speed constraint (an overtaking vessel must be materially
# faster than the vessel overtaken) is enforced by REPAIR in the Python
# bridge (see _scene_to_spec below), not by a `require` clause here. A
# `require` referencing globalParameters raised a NameError under the
# Scenic version this project targets, and rejection sampling would in any
# case make the acceptance rate depend on the encounter type and leave the
# search space disconnected -- repair keeps every draw usable.

# ---- a minimal concrete scene so the sample is inspectable in Scenic ----
class Vessel:
    width: 24
    length: 150
    allowCollisions: True
    requireVisible: False

ownShip = new Vessel at (0, 0), facing 0 deg
ego = ownShip
targetShip = new Vessel at (0, 3000), facing 180 deg
