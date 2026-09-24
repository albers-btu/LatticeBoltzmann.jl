const TYPE_S  = 0x01 # (S)olid / bounce-back
const TYPE_E  = 0x02 # (E)quilibrium / inflow-outflow
const TYPE_T  = 0x04 # (T)emperature
const TYPE_F  = 0x08 # (F)luid
const TYPE_I  = 0x10 # (I)nterface
const TYPE_G  = 0x20 # (G)as
const TYPE_H  = 0x40 # (H)eat-flux

const TYPE_MS = 0x03 # S|E    (M)oving (S)olid
const TYPE_BO = 0x03 # S|E    (BO)undary, Solid or Equilibrium
const TYPE_IF = 0x18 # I|F    (I)nterface to (F)luid
const TYPE_IG = 0x30 # I|G    (I)nterface to (G)as
const TYPE_GI = 0x38 # F|I|G  (G)as       to (I)nterface
const TYPE_SU = 0x38 # F|I|G  (SU)rface