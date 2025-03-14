// Aim.h
#pragma once

#include <vector>
#include <optional>
#include <chrono>

struct Detection;

struct TargetEntry {
    float x;
    float y;
    std::chrono::steady_clock::time_point last_update;
};

class Aim
{
public:
    Aim(unsigned int window_width, unsigned int window_height);

    void update_targets(const std::vector<Detection>& detections, bool aim_active);

    void set_target_lifetime(std::chrono::milliseconds lifetime) { target_lifetime = lifetime; }

private:
    unsigned int window_width, window_height;
    float origin_x, origin_y;

    std::vector<TargetEntry> targets;

    // Locked target position
    std::optional<TargetEntry> locked_target;
    // Keep track of how many consecutive frames we've lost the locked target
    int lostFrames = 0;
    int lostFramesAllowed = 5; // grace period frames

    // Lifetimes and thresholds
    std::chrono::milliseconds target_lifetime = std::chrono::milliseconds(200);
    // Define thresholds and factors
    const float convergence_threshold = 1.0f;       // Distance within which to stop
    const float no_smoothing_threshold = 50.0f;     // Distance below which to snap
    const float mouse_smoothing_factor = 0.7f;      // Base smoothing factor
    const float max_step = 50.0f;                   // Maximum pixels to move per update
    const float min_step = 1.0f;                    // Minimum pixels to move to prevent jitter


    void initialize_mouse_position();

    float distance(float x1, float y1, float x2, float y2);
    bool is_target_similar(const TargetEntry& existing, float x, float y, float tolerance);
    void add_or_update_target(float x, float y);
    void remove_stale_targets();

    std::optional<TargetEntry> find_closest_target_to_screen_center();
    void lock_onto_target(const TargetEntry& t);

    void aim_at_target(const TargetEntry& t, bool aim_active);
    void maintain_lost_target_aim(bool aim_active);
    void move_mouse_towards(float target_x, float target_y, bool aim_active);
};