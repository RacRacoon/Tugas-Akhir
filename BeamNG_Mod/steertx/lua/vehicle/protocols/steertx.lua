-- steertx.lua
-- Custom UDP protocol: kirim steering input (BeamNG tidak expose ini lewat
-- OutGauge maupun MotionSim). Dibuat mengikuti struktur resmi outGauge.lua
-- (lua/vehicle/protocols/outGauge.lua bawaan game), lihat https://go.beamng.com/protocols
--
-- INSTALASI:
--   Taruh folder ini (steertx/) di dalam:
--     <USER_FOLDER>/mods/unpacked/steertx/
--   sehingga struktur akhirnya:
--     <USER_FOLDER>/mods/unpacked/steertx/lua/vehicle/protocols/steertx.lua
--   <USER_FOLDER> bisa dicek di game: Options > General > tombol "Open User Folder".
--   Setelah file ditaruh, masuk ke game, load mobil, tekan Ctrl+R (reload vehicle Lua).
--   Protokol "steertx" akan muncul di Options > Other > Protocols — aktifkan,
--   set IP (127.0.0.1) dan PORT (bebas, jangan bentrok dengan OutGauge/MotionSim,
--   misal 4124).
--
-- CATATAN: electrics.values TIDAK didokumentasikan resmi field-per-field, jadi
-- dikirim 2 kandidat (steering & steering_input) sekaligus. Pakai mode --probe
-- di telext_v2.py buat lihat mana yang beneran berubah saat setir diputar
-- di game, lalu set STEER_FIELD di telext_v2.py sesuai itu.

local M = {}

local function init() end
local function reset() end
local function getAddress()        return settings.getValue("protocols_steertx_address") end
local function getPort()           return settings.getValue("protocols_steertx_port") end
local function getMaxUpdateRate()  return settings.getValue("protocols_steertx_maxUpdateRate") end

local function isPhysicsStepUsed()
  return false -- graphics step cukup, sama seperti outGauge.lua
end

local function getStructDefinition()
  return [[
    //////////////////////////////////////////////////////////////////////////////////
    ////// steertx — protokol custom, bukan bagian dari protokol resmi BeamNG //////////
    //////////////////////////////////////////////////////////////////////////////////
    unsigned time;           // N/A, selalu 0 (placeholder, ikut pola outGauge.lua)
    float    steering;       // kandidat 1: electrics.values.steering
    float    steeringInput;  // kandidat 2: electrics.values.steering_input
    float    wheelspeed;     // referensi silang ke electrics.values.wheelspeed (cek sinkron dgn OutGauge)
  ]]
end

local function fillStruct(o, dtSim)
  if not electrics.values.watertemp then
    -- kendaraan belum selesai inisialisasi, jangan kirim dulu
    return
  end

  o.time          = 0
  o.steering      = electrics.values.steering or 0
  o.steeringInput = electrics.values.steering_input or electrics.values.steeringInput or 0
  o.wheelspeed    = electrics.values.wheelspeed or 0
end

M.init = init
M.reset = reset
M.getAddress = getAddress
M.getPort = getPort
M.getMaxUpdateRate = getMaxUpdateRate
M.getStructDefinition = getStructDefinition
M.fillStruct = fillStruct
M.isPhysicsStepUsed = isPhysicsStepUsed

return M
