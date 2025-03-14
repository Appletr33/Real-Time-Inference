// Aim.cpp
#include "Aim.h"
#include <Windows.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <thread>

#include "YOLOv11.h"


static inline float ToNDC_X(float screen_x, float screen_width) {
    return (2.0f * screen_x) / screen_width - 1.0f;
}

static inline float ToNDC_Y(float screen_y, float screen_height) {
    return 1.0f - (2.0f * screen_y) / screen_height;
}

Aim::Aim(unsigned int window_width, unsigned int window_height)
    : window_width(window_width), window_height(window_height)
{
    // Adjust if needed; here we assume a known origin offset
    origin_x = (window_width / 2.0f) - 320.0f;
    origin_y = (window_height / 2.0f) - 320.0f;
    initialize_mouse_position();
}

void Aim::initialize_mouse_position()
{
    // Not strictly needed for this logic, but left as placeholder
}

float Aim::distance(float x1, float y1, float x2, float y2)
{
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

bool Aim::is_target_similar(const TargetEntry& existing, float x, float y, float tolerance)
{
    float dist = distance(existing.x, existing.y, x, y);
    return dist <= tolerance;
}

void Aim::add_or_update_target(float x, float y)
{
    float similarity_tolerance = 50.0f;
    auto now = std::chrono::steady_clock::now();

    // Find similar target
    for (auto& t : targets) {
        if (is_target_similar(t, x, y, similarity_tolerance)) {
            // Update position, no smoothing here for the target itself
            // The locked target handling is done elsewhere
            t.x = x;
            t.y = y;
            t.last_update = now;
            return;
        }
    }

    // New target
    TargetEntry new_t;
    new_t.x = x;
    new_t.y = y;
    new_t.last_update = now;
    targets.push_back(new_t);
}

void Aim::remove_stale_targets()
{
    auto now = std::chrono::steady_clock::now();
    targets.erase(std::remove_if(targets.begin(), targets.end(), [&](const TargetEntry& t) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - t.last_update);
        return elapsed > target_lifetime;
        }), targets.end());
}

std::optional<TargetEntry> Aim::find_closest_target_to_screen_center()
{
    if (targets.empty())
        return std::nullopt;

    float center_x = window_width * 0.5f;
    float center_y = window_height * 0.5f;

    float best_dist = std::numeric_limits<float>::max();
    const TargetEntry* best_t = nullptr;

    for (auto& t : targets) {
        float d = distance(center_x, center_y, t.x, t.y);
        if (d < best_dist) {
            best_dist = d;
            best_t = &t;
        }
    }

    if (best_t)
        return *best_t;
    return std::nullopt;
}

void Aim::lock_onto_target(const TargetEntry& t)
{
    locked_target = t;
    lostFrames = 0;
}

void Aim::aim_at_target(const TargetEntry& t, bool aim_active)
{
    move_mouse_towards(t.x, t.y, aim_active);
}

void Aim::maintain_lost_target_aim(bool aim_active)
{
    if (!locked_target.has_value())
        return;

    // If we lost the target but haven't exceeded lostFramesAllowed, keep aiming at the last known position
    lostFrames++;
    if (lostFrames <= lostFramesAllowed) {
        // Keep aiming at locked_target's last known position
        move_mouse_towards(locked_target->x, locked_target->y, aim_active);
    }
    else {
        // Lost too long, give up
        locked_target.reset();
        lostFrames = 0;
        // No movement if no targets
    }
}

void Aim::move_mouse_towards(float target_x, float target_y, bool aim_active)
{
    if (!aim_active)
        return;

    POINT cursorPos;
    if (!GetCursorPos(&cursorPos))
        return;

    float currentX = static_cast<float>(cursorPos.x);
    float currentY = static_cast<float>(cursorPos.y);

    float d = distance(currentX, currentY, target_x, target_y);
    if (d <= convergence_threshold) {
        // Close enough, don't move to avoid jitter
        return;
    }

    float dx = target_x - currentX;
    float dy = target_y - currentY;

    // Normalize the direction vector
    float dir_x = dx / d;
    float dir_y = dy / d;

    float step;

    if (d < no_smoothing_threshold) {
        // When close to the target, reduce the step size proportionally
        // to avoid overshooting
        step = std::max(d * mouse_smoothing_factor, min_step);
    }
    else {
        // When far, use the smoothing factor but cap the step size
        step = std::max(std::min(d * mouse_smoothing_factor, max_step), min_step);
    }

    // Calculate movement deltas
    float move_x = dir_x * step;
    float move_y = dir_y * step;

    // Convert to int for mouse_event
    int moveX = static_cast<int>(std::round(move_x));
    int moveY = static_cast<int>(std::round(move_y));

    // Apply mouse movement
    mouse_event(MOUSEEVENTF_MOVE, moveX, moveY, 0, 0);
}

void Aim::update_targets(const std::vector<Detection>& detections, bool aim_active)
{
    // 1. Update target list
    for (const auto& detection : detections) {

        float sx_min = origin_x + detection.bbox.x;
        float sy_min = origin_y + detection.bbox.y;
        float sx_max = sx_min + detection.bbox.width;
        float sy_max = sy_min + detection.bbox.height;

        float mid_x, mid_y;
        if (detection.bbox.area() > 70.0f) 
        {
            // Adjusted x position: midpoint shifted 7% to the left
            mid_x = (sx_min + sx_max) * 0.5f - detection.bbox.width * 0.07f;
            // Adjusted y position: 85% of the box height
            mid_y = sy_min + detection.bbox.height * 0.15f;
        }
        else
        {
            // Adjusted x position: midpoint shifted 5% to the left
            mid_x = (sx_min + sx_max) * 0.5f - detection.bbox.width * 0.05f;
            // Adjusted y position: 75% of the box height
            mid_y = sy_min + detection.bbox.height * 0.25f;
        }
        add_or_update_target(mid_x, mid_y);
    }
    // 2. Remove stale targets
    remove_stale_targets();

    // Logic:
    // If we have a locked target, check if it's still in the targets list (or close)
    // If yes, aim at it. If not, try lost target logic.
    // If no locked target, try to lock onto a new one if available.
    // If no targets and no locked target, do nothing.

    if (locked_target.has_value()) {
        // Check if locked target still matches something in the list
        // We'll try to find a target similar to the locked target position
        float similarity_tolerance = 50.0f;
        bool found_locked = false;
        for (auto& t : targets) {
            if (is_target_similar(t, locked_target->x, locked_target->y, similarity_tolerance)) {
                // Re-lock onto updated position
                lock_onto_target(t);
                found_locked = true;
                break;
            }
        }

        if (found_locked) {
            aim_at_target(*locked_target, aim_active);
        }
        else {
            // Locked target not found this frame
            maintain_lost_target_aim(aim_active);
        }
    }
    else {
        // No locked target
        if (!targets.empty()) {
            // Lock onto closest
            auto closest = find_closest_target_to_screen_center();
            if (closest.has_value()) {
                lock_onto_target(closest.value());
                aim_at_target(*locked_target, aim_active);
            }
            else {
                // This theoretically shouldn't happen since we have targets
                // but no valid closest. Just do nothing
            }
        }
        else {
            // No targets, no locked target, do nothing (no movement)
        }
    }
}