"""HID++ 2.0 feature IDs and error codes used by ambiance."""

ROOT = 0x0000
FEATURE_SET = 0x0001
DEVICE_FW_VERSION = 0x0003
DEVICE_NAME = 0x0005
CONFIG_CHANGE = 0x0020  # NB: SetComplete(cookie=0) releases the G515's SW lighting claim
UNIFIED_BATTERY = 0x1004
WIRELESS_DEVICE_STATUS = 0x1D4B
BRIGHTNESS_CONTROL = 0x8040
COLOR_LED_EFFECTS = 0x8070
RGB_EFFECTS = 0x8071
PER_KEY_LIGHTING_V2 = 0x8081
ONBOARD_PROFILES = 0x8100
PROFILE_MANAGEMENT = 0x8101

ERRORS = {
    0x00: "NoError",
    0x01: "Unknown",
    0x02: "InvalidArgument",
    0x03: "OutOfRange",
    0x04: "HardwareError",
    0x05: "LogitechInternal",
    0x06: "InvalidFeatureIndex",
    0x07: "InvalidFunctionId",
    0x08: "Busy",
    0x09: "Unsupported",
}
ERR_BUSY = 0x08
