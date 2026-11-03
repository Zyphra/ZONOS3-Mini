cmake_minimum_required(VERSION 3.18)
project(zonos3-cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- zonos3-app (desktop shell) opt-in. Declared here, before add_subdirectory(ggml),
#     specifically so the Apple Objective-C / -C++ languages it needs are enabled in
#     THIS root scope first. saucer's WKWebView backend + saucer-embed inject .m/.mm
#     sources into the top-level zonos3-app target; ggml's Metal backend also enables
#     OBJC — but inside ggml's own subdirectory scope. Whichever scope enables a
#     language *first* is the only one that gets CMAKE_<LANG>_COMPILE_OBJECT, so if
#     ggml (a subdir) wins the race the top-level target is left without the rule and
#     CMake dies at generate with "CMAKE_OBJC_COMPILE_OBJECT not set". Enabling both
#     here, pre-ggml, makes the root scope the first enabler and the rules propagate
#     down to ggml and up to zonos3-app alike. C++23 toolchain still required (README). ---
option(ZONOS3_APP "Build the zonos3-app desktop shell (requires a C++23 toolchain, see README)" OFF)
if (ZONOS3_APP AND APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

# --- CUDA target (H100 = sm_90). Enable with -DGGML_CUDA=ON ---
if (GGML_CUDA)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "" FORCE)
    endif()
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE) # CUDA-graph replay for real-time decode
endif()

# --- ggml (upstream submodule, pinned to a pristine SHA) ---
# We carry local ggml fixes as patches under patches/ rather than forking the
# submodule. They are (re)applied idempotently at configure time: the reverse
# --check skips an already-patched tree, so re-running cmake is a no-op. If a
# future submodule bump makes a patch no longer apply, configure fails loudly.
find_package(Git REQUIRED)
file(GLOB GGML_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch)
list(SORT GGML_PATCHES)
foreach(_patch ${GGML_PATCHES})
    get_filename_component(_name ${_patch} NAME)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/ggml apply --reverse --check ${_patch}
        RESULT_VARIABLE _applied OUTPUT_QUIET ERROR_QUIET)
    if (_applied EQUAL 0)
        message(STATUS "ggml patch already applied: ${_name}")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR}/ggml apply ${_patch}
            RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if (_rc EQUAL 0)
            message(STATUS "ggml patch applied: ${_name}")
        else()
            message(FATAL_ERROR "Failed to apply ggml patch ${_name}:\n${_err}")
        endif()
    endif()
endforeach()

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

# The CPU DAC calls Accelerate's sgemm directly (src/dac.cpp) so each conv accumulates its taps,
# bias and residual into one output buffer. Every target compiling dac.cpp uses zonos3_use_dac().
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
endif()
function(zonos3_use_dac target)
    if (APPLE)
        target_link_libraries(${target} PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()
endfunction()

# Backbone and native audio paths are shared by CLI, server, and desktop.
add_library(zonos3 STATIC
    src/zonos3.cpp src/zonos3-graph.cpp src/zonos3-sampler.cpp
    src/zonos3-prompt.cpp src/zonos3-tokenizer.cpp src/zonos3-steer.cpp src/zonos3-markup.cpp)
target_include_directories(zonos3 PUBLIC src ggml/include PRIVATE vendor)
target_link_libraries(zonos3 PUBLIC ggml)

add_library(zonos3-audio STATIC
    src/dac.cpp src/dac-encoder.cpp src/ssl-encoder.cpp src/spk-encoder.cpp src/audio-decode.cpp)
target_include_directories(zonos3-audio PUBLIC src ggml/include PRIVATE vendor)
target_link_libraries(zonos3-audio PUBLIC ggml)
zonos3_use_dac(zonos3-audio)

# --- Pronunciation lexicon (plan 7.9): vendored LMDB 0.9.36 (OpenLDAP libraries/liblmdb, tag LMDB_0.9.36)
#     and the phonemes.py port. espeak-ng is dlopen'ed at run time (ZONOS3_ESPEAK_NG_LIBRARY), never linked. ---
find_package(Threads REQUIRED)
add_library(zonos3-lmdb STATIC vendor/lmdb/mdb.c vendor/lmdb/midl.c)
target_include_directories(zonos3-lmdb PUBLIC vendor/lmdb)
target_link_libraries(zonos3-lmdb PUBLIC Threads::Threads)
if (MSVC)
    target_compile_options(zonos3-lmdb PRIVATE /w)
else()
    target_compile_options(zonos3-lmdb PRIVATE -w)
endif()
target_sources(zonos3 PRIVATE src/zonos3-phonemes.cpp)
target_link_libraries(zonos3 PUBLIC zonos3-lmdb ${CMAKE_DL_LIBS})
# --- end pronunciation lexicon ---

# --- Dub planning (plan 7.12): port of model/src/zonos3/dub/plan.py ---
target_sources(zonos3 PRIVATE src/dub-plan.cpp src/align-text.cpp)
# --- end dub planning ---

# --- LACI and span-follow (plan 7.8): ports of model/src/zonos3/tts/{laci,span_follow}.py ---
target_sources(zonos3 PRIVATE src/zonos3-laci.cpp src/zonos3-span-follow.cpp)
# --- end LACI and span-follow ---

# --- Graph features (plan 7.5/7.6/7.8): accent adapters, emotion directions; the alignment probe lives in the graph ---
target_sources(zonos3 PRIVATE src/zonos3-accent.cpp src/zonos3-emotion.cpp)
# --- end graph features ---

# --- Speech editing (plan 7.11): port of model/src/zonos3/tts/edit.py, the edit_align.py planning and textmatch.py ---
target_sources(zonos3 PRIVATE src/zonos3-edit.cpp src/zonos3-textmatch.cpp)
# --- end speech editing ---

add_executable(zonos3-cli src/main.cpp)
target_link_libraries(zonos3-cli PRIVATE zonos3 zonos3-audio)
target_include_directories(zonos3-cli PRIVATE vendor)

add_executable(dac-cli src/dac-cli.cpp)
target_link_libraries(dac-cli PRIVATE zonos3-audio)
add_executable(dac-encoder-cli src/dac-encoder-cli.cpp)
target_link_libraries(dac-encoder-cli PRIVATE zonos3-audio)
add_executable(ssl-encoder-cli src/ssl-encoder-cli.cpp)
target_link_libraries(ssl-encoder-cli PRIVATE zonos3-audio)
add_executable(spk-encoder-cli src/spk-encoder-cli.cpp)
target_link_libraries(spk-encoder-cli PRIVATE zonos3-audio)

add_executable(ffmpeg-decode-smoke EXCLUDE_FROM_ALL
    tests/ffmpeg-decode-smoke.cpp src/audio-decode.cpp)
target_include_directories(ffmpeg-decode-smoke PRIVATE src)
if (WIN32)
    add_executable(fake-ffmpeg EXCLUDE_FROM_ALL tests/fake-ffmpeg.cpp)
endif()

add_executable(quantize-cli src/quantize.cpp)
target_include_directories(quantize-cli PRIVATE src ggml/include)
target_link_libraries(quantize-cli PRIVATE ggml)
add_executable(prune-cli src/prune.cpp)
target_include_directories(prune-cli PRIVATE src ggml/include)
target_link_libraries(prune-cli PRIVATE ggml)
add_executable(zonos3-perplexity src/perplexity.cpp)
target_link_libraries(zonos3-perplexity PRIVATE zonos3)
target_include_directories(zonos3-perplexity PRIVATE vendor)

# --- HTTP server (plan 7.1): src/server/*, the API spec embedded for request validation ---
# Request validation: vendored pboettch/json-schema-validator 2.3.0 (MIT) against api/openapi.json, embedded as a
# generated header (re-generated whenever the spec changes).
set(ZONOS3_OPENAPI ${CMAKE_CURRENT_SOURCE_DIR}/../../api/openapi.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ZONOS3_OPENAPI})
file(READ ${ZONOS3_OPENAPI} ZONOS3_OPENAPI_HEX HEX)
string(LENGTH "${ZONOS3_OPENAPI_HEX}" ZONOS3_OPENAPI_HEX_LEN)
math(EXPR ZONOS3_OPENAPI_LEN "${ZONOS3_OPENAPI_HEX_LEN} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ZONOS3_OPENAPI_BYTES "${ZONOS3_OPENAPI_HEX}")
configure_file(src/server/openapi.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/zonos3-openapi.h @ONLY)
# The validator includes <nlohmann/json.hpp>: point it at the vendored single header.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/nlohmann/json.hpp "#include \"${CMAKE_CURRENT_SOURCE_DIR}/vendor/json.hpp\"\n")
set(JSV vendor/json-schema-validator/src)
add_library(zonos3-json-schema STATIC
    ${JSV}/json-validator.cpp ${JSV}/json-uri.cpp ${JSV}/json-patch.cpp ${JSV}/json-schema-draft7.json.cpp
    ${JSV}/string-format-check.cpp ${JSV}/smtp-address-validator.cpp)
target_include_directories(zonos3-json-schema PUBLIC ${JSV} ${CMAKE_CURRENT_BINARY_DIR}/generated)
if (MSVC)
    target_compile_options(zonos3-json-schema PRIVATE /w)
else()
    target_compile_options(zonos3-json-schema PRIVATE -w)
endif()

# Everything of the server but main(): zonos3-server and the embedding zonos3-app share it.
add_library(zonos3-server-core STATIC
    src/server/http.cpp src/server/schema.cpp src/server/util.cpp src/server/workers.cpp src/server/textnorm.cpp src/server/fusealign.cpp
    src/server/asr.cpp src/server/edit_align.cpp
    src/server/scheduler.cpp src/server/speakers.cpp src/server/generate.cpp src/server/capabilities.cpp
    src/server/projects.cpp src/server/phonemize.cpp src/server/edit.cpp src/server/dub.cpp)
target_include_directories(zonos3-server-core PUBLIC src vendor ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(zonos3-server-core PUBLIC zonos3 zonos3-audio zonos3-json-schema Threads::Threads)
if (WIN32)
    target_link_libraries(zonos3-server-core PUBLIC ws2_32)
endif()
add_executable(zonos3-server src/server/main.cpp)
target_link_libraries(zonos3-server PRIVATE zonos3-server-core)
# --- end HTTP server ---

# --- zonos3-cli --request (plan 7.14): the server's request resolver builds the prompt ---
target_link_libraries(zonos3-cli PRIVATE zonos3-server-core)
# --- end zonos3-cli --request ---

include(CTest)
if (BUILD_TESTING)
    add_executable(raw-reference-state tests/raw-reference-state.cpp)
    target_link_libraries(raw-reference-state PRIVATE zonos3)
    add_test(NAME raw-reference-state COMMAND raw-reference-state
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dummy-zonos3.gguf)
    add_executable(steering tests/steering.cpp)
    target_link_libraries(steering PRIVATE zonos3)
    add_test(NAME steering COMMAND steering
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dummy-zonos3.gguf
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dummy-steer.gguf)
    set_tests_properties(steering PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
    # Streamed (block 3) == buffered: iSTFT overlap-add / FIR mixer holdback and flush.
    add_test(NAME dac-seam COMMAND dac-cli --seam tests/fixtures/dummy-dac.gguf
        tests/fixtures/dummy-dac-codes.npy --block 3 --cpu
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(dac-seam PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
    add_executable(prune-policy tests/prune-policy.cpp)
    target_include_directories(prune-policy PRIVATE src)
    add_test(NAME prune-policy COMMAND prune-policy)
    add_executable(ssl-reference-state tests/ssl-reference-state.cpp)
    target_link_libraries(ssl-reference-state PRIVATE zonos3-audio)
    set(ZONOS3_TEST_SSL_TEACHER "" CACHE FILEPATH "Frozen teacher GGUF for the optional real-weight lifecycle regression")
    if (ZONOS3_TEST_SSL_TEACHER)
        add_test(NAME ssl-reference-state COMMAND ssl-reference-state ${ZONOS3_TEST_SSL_TEACHER})
    endif()
    find_program(BASH_PROGRAM bash)
    if (BASH_PROGRAM)
        add_test(NAME tokenizer COMMAND ${BASH_PROGRAM} tests/tokenizer.sh $<TARGET_FILE:zonos3-cli>)
        add_test(NAME evaluation-tools COMMAND ${BASH_PROGRAM} tests/evaluation-tools.sh
            $<TARGET_FILE:zonos3-cli> $<TARGET_FILE:zonos3-perplexity>
            $<TARGET_FILE:quantize-cli> $<TARGET_FILE:prune-cli>)
        set_tests_properties(tokenizer evaluation-tools PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endif()

# --- Graph features (plan 7.5/7.6/7.8): CPU invariants of the alignment probe, accent adapters and emotion directions;
#     also the emotion parity driver of tests/parity/test_ggml_graph.py. Fixtures: scripts/make-dummy-{accents,directions}.py ---
if (BUILD_TESTING)
    add_executable(graph-features tests/graph-features.cpp)
    target_link_libraries(graph-features PRIVATE zonos3)
    target_include_directories(graph-features PRIVATE vendor)
    add_test(NAME graph-features COMMAND graph-features --selftest
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dummy-zonos3.gguf
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dummy-accents.gguf
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dummy-emotion.gguf)
    set_tests_properties(graph-features PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
endif()
# --- end graph features tests ---

# --- LACI replay (plan 7.14): parity driver for tests/parity/test_ggml_laci_replay.py; the ctest cases replay
#     tests/fixtures/laci-replay/<case>.trace.json against <case>.expected.json (written by the Python driver) ---
if (BUILD_TESTING)
    add_executable(laci-replay tests/laci-replay.cpp)
    target_link_libraries(laci-replay PRIVATE zonos3)
    target_include_directories(laci-replay PRIVATE vendor)
    file(GLOB ZONOS3_LACI_REPLAY_CASES ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/laci-replay/*.trace.json)
    foreach (trace ${ZONOS3_LACI_REPLAY_CASES})
        string(REGEX REPLACE "\\.trace\\.json$" "" case_path ${trace})
        get_filename_component(case ${case_path} NAME)
        add_test(NAME laci-replay-${case} COMMAND laci-replay --trace ${trace} --expect ${case_path}.expected.json)
        set_tests_properties(laci-replay-${case} PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
    endforeach()
endif()
# --- end LACI replay ---

# --- Dub planning replay (plan 7.14): parity driver for tests/parity/test_ggml_dub_plan.py ---
if (BUILD_TESTING)
    add_executable(dub-plan-replay tests/dub-plan-replay.cpp)
    target_link_libraries(dub-plan-replay PRIVATE zonos3)
    target_include_directories(dub-plan-replay PRIVATE vendor)
    add_test(NAME dub-plan-replay COMMAND dub-plan-replay --smoke)
    set_tests_properties(dub-plan-replay PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
endif()
# --- end dub planning replay ---

# --- Speech-edit replay (plan 7.11): parity driver for tests/parity/test_ggml_edit.py; the ctest case replays
#     tests/fixtures/edit-replay/cases.json (Python answers written by that test) ---
if (BUILD_TESTING)
    add_executable(edit-replay tests/edit-replay.cpp)
    target_link_libraries(edit-replay PRIVATE zonos3)
    target_include_directories(edit-replay PRIVATE vendor)
    add_test(NAME edit-replay COMMAND edit-replay --cases ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/edit-replay/cases.json)
    set_tests_properties(edit-replay PROPERTIES PASS_REGULAR_EXPRESSION "PASS")
endif()
# --- end speech-edit replay ---

# --- Desktop app: the server embedded in a native webview window (saucer). Opt-in
#     because saucer needs a much newer toolchain than the rest of the project:
#     CMake >= 3.25, C++23 (GCC >= 14 / Clang >= 20 / AppleClang >= 17 / MSVC >= 19.44),
#     and on Linux GTK4 >= 4.12 + WebKitGTK-6.0 + libadwaita + json-glib dev packages.
#     Everything else in this file keeps building with the C++17 / CMake 3.18 floor. ---
# ZONOS3_APP is declared (and its Apple languages enabled) near the top of this file,
# before add_subdirectory(ggml); see the comment there for why the ordering matters.
if (ZONOS3_APP)
    if (CMAKE_VERSION VERSION_LESS 3.31)
        # saucer core says 3.25, but its saucer-embed dependency uses policy CMP0174 (3.31).
        message(FATAL_ERROR "ZONOS3_APP requires CMake >= 3.31 (saucer); this is CMake ${CMAKE_VERSION}")
    endif()

    include(FetchContent)
    FetchContent_Declare(saucer
        GIT_REPOSITORY https://github.com/saucer/saucer
        GIT_TAG        v8.0.5)
    FetchContent_Declare(saucer-desktop      # native file pickers (saucer module)
        GIT_REPOSITORY https://github.com/saucer/desktop
        GIT_TAG        v4.1.0)
    FetchContent_MakeAvailable(saucer saucer-desktop)

    add_executable(zonos3-app src/app.cpp src/server/main.cpp)
    target_compile_definitions(zonos3-app PRIVATE ZONOS3_APP_EMBED)   # server/main.cpp: no main()
    target_compile_features(zonos3-app PRIVATE cxx_std_23)
    target_include_directories(zonos3-app PRIVATE src vendor ggml/include)
    target_link_libraries(zonos3-app PRIVATE zonos3-server-core saucer::saucer saucer::desktop Threads::Threads)

    if (WIN32)
        # GUI subsystem (no console window) with the icon resource; saucer has no
        # WinMain, so keep the standard main() as the entry point.
        target_sources(zonos3-app PRIVATE assets/app/zonos3.rc)
        set_target_properties(zonos3-app PROPERTIES WIN32_EXECUTABLE TRUE)
        target_link_libraries(zonos3-app PRIVATE ws2_32)
        if (MSVC)
            target_link_options(zonos3-app PRIVATE "/ENTRY:mainCRTStartup")
        endif()
    elseif (APPLE)
        # Zonos3.app bundle: the studio (studio/dist of the repo), default voices, the Python packages of the helper
        # workers (model/src/zonos3 and fusealign_hub of the third_party/fusealign-multilingual submodule) and the icon
        # ship inside Contents/Resources (app.cpp and the server resolve ../Resources/{studio,python} relative to the
        # executable).
        set(ZONOS3_APP_RESOURCES assets/app/zonos3.icns assets/app/zonos3.png)
        target_sources(zonos3-app PRIVATE ${ZONOS3_APP_RESOURCES})
        set_source_files_properties(${ZONOS3_APP_RESOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
        set(ZONOS3_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
        file(GLOB ZONOS3_VOICE_RESOURCES ${ZONOS3_REPO_ROOT}/default_voices/*)
        target_sources(zonos3-app PRIVATE ${ZONOS3_VOICE_RESOURCES})
        set_source_files_properties(${ZONOS3_VOICE_RESOURCES} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/default_voices)
        file(GLOB_RECURSE ZONOS3_PYTHON_RESOURCES ${ZONOS3_REPO_ROOT}/model/src/zonos3/*)
        list(FILTER ZONOS3_PYTHON_RESOURCES EXCLUDE REGEX "/__pycache__/")
        target_sources(zonos3-app PRIVATE ${ZONOS3_PYTHON_RESOURCES})
        foreach(res ${ZONOS3_PYTHON_RESOURCES})
            file(RELATIVE_PATH rel ${ZONOS3_REPO_ROOT}/model/src ${res})
            get_filename_component(reldir ${rel} DIRECTORY)
            set_source_files_properties(${res} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/python/${reldir})
        endforeach()
        set(ZONOS3_FUSEALIGN_SRC ${ZONOS3_REPO_ROOT}/third_party/fusealign-multilingual)
        if (NOT EXISTS ${ZONOS3_FUSEALIGN_SRC}/fusealign_hub/__init__.py)
            message(FATAL_ERROR "ZONOS3_APP bundles FuseAlign: git submodule update --init third_party/fusealign-multilingual")
        endif()
        file(GLOB_RECURSE ZONOS3_FUSEALIGN_RESOURCES ${ZONOS3_FUSEALIGN_SRC}/fusealign_hub/*.py)
        target_sources(zonos3-app PRIVATE ${ZONOS3_FUSEALIGN_RESOURCES} ${ZONOS3_FUSEALIGN_SRC}/LICENSE)
        set_source_files_properties(${ZONOS3_FUSEALIGN_RESOURCES} ${ZONOS3_FUSEALIGN_SRC}/LICENSE
            PROPERTIES MACOSX_PACKAGE_LOCATION Resources/python/fusealign_hub)
        file(GLOB_RECURSE ZONOS3_STUDIO_RESOURCES ${ZONOS3_REPO_ROOT}/studio/dist/*)
        target_sources(zonos3-app PRIVATE ${ZONOS3_STUDIO_RESOURCES})
        foreach(res ${ZONOS3_STUDIO_RESOURCES})
            file(RELATIVE_PATH rel ${ZONOS3_REPO_ROOT}/studio/dist ${res})
            get_filename_component(reldir ${rel} DIRECTORY)
            set_source_files_properties(${res} PROPERTIES MACOSX_PACKAGE_LOCATION Resources/studio/${reldir})
        endforeach()
        set_target_properties(zonos3-app PROPERTIES
            OUTPUT_NAME Zonos3
            MACOSX_BUNDLE TRUE
            MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/assets/app/Info.plist.in
            MACOSX_BUNDLE_BUNDLE_NAME Zonos3
            MACOSX_BUNDLE_GUI_IDENTIFIER cpp.zonos3.app
            MACOSX_BUNDLE_BUNDLE_VERSION 1.0
            MACOSX_BUNDLE_SHORT_VERSION_STRING 1.0
            MACOSX_BUNDLE_ICON_FILE zonos3.icns)
    endif()
endif()
