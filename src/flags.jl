const TYPE_S  = 0x01 # solid (bounce-back)
const TYPE_E  = 0x02 # equilibrium / inflow-outflow
const TYPE_T  = 0x04 # temperature wall
const TYPE_F  = 0x08 # fluid (free-surface)
const TYPE_I  = 0x10 # interface
const TYPE_G  = 0x20 # gas
const TYPE_MS = 0x03 # next to moving solid

const TYPE_BO = 0x03 # S|E
const TYPE_IF = 0x18 # I|F    interface -> fluid
const TYPE_IG = 0x30 # I|G    interface -> gas
const TYPE_GI = 0x38 # F|I|G  gas -> interface
const TYPE_SU = 0x38 # mask of F|I|G