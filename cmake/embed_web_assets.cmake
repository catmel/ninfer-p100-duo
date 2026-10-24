# Embeds every src/serve/web file into web_assets.cpp as byte arrays.
# Invoked as a build-time script command:
#   cmake -DINPUT_DIR=... -DTEMPLATE=... -DOUTPUT=... -DASSETS="index.html;app.js" -P embed_web_assets.cmake
# Re-runs whenever an asset changes through the MAIN_DEPENDENCY of the custom command plus
# the directory listing captured at configure time.

set(ASSET_ARRAYS "")
set(ASSET_ROWS "")

foreach(rel IN LISTS ASSETS)
  file(READ "${INPUT_DIR}/${rel}" hex HEX)
  if(hex STREQUAL "")
    set(hex_bytes "0")
  else()
    # One "0xNN," pair per byte; 32 pairs (160 chars) per line keeps the array readable.
    string(REGEX REPLACE "(..)" "0x\\1," hex_pairs "${hex}")
    string(REGEX REPLACE "(.{160})" "\\1\n    " hex_bytes "${hex_pairs}")
  endif()
  string(REGEX REPLACE "[^A-Za-z0-9]" "_" array_name "${rel}")
  string(APPEND ASSET_ARRAYS
    "constexpr std::uint8_t k_asset_${array_name}[] = {\n    ${hex_bytes}};\n")
  string(APPEND ASSET_ROWS
    "        WebAsset{\"${rel}\", "
    "{reinterpret_cast<const char*>(k_asset_${array_name}), sizeof(k_asset_${array_name})}},\n")
endforeach()

configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)
