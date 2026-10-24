#pragma once

// The embedded web panel (src/serve/web). web_assets.cpp is generated at build time by
// cmake/embed_web_assets.cmake so the server binary is self-contained.

#include <string_view>
#include <vector>

namespace ninfer::serve::web {

struct WebAsset {
    std::string_view path;       // relative to "web/", e.g. "app.js" or "fonts/outfit-latin-wght.woff2"
    std::string_view content;
};

// All embedded assets; "index.html" is the panel entry.
const std::vector<WebAsset>& assets();

} // namespace ninfer::serve::web
