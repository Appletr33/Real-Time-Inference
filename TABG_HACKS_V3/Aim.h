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
    std::chrono::milliseconds target_lifetime = std::chrono::milliseconds(100);
    float mouse_smoothing_factor = 0.15f; // fraction of distance to move each frame when far
    float no_smoothing_threshold = 40.0f; // if closer than this distance to the target, snap directly
    float convergence_threshold = 1.0f;   // if closer than this to the target, might not need to move

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