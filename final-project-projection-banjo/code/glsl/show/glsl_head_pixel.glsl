// glsl_head_pixel.glsl  - content painted onto the banjo head (square texture)
//
// uA = (time s, scene, intensity)
// uB = (hue shift 0..1, audio 0..1, edge feather 0..0.3)
// uC = (head-plane velocity x, y  [head diameters / s], flame lean)
// uD = (visible 0..1, world-up x, world-up y)   <- up is in head-local coords
//
// Scenes (uA.y, fractional values crossfade):
//   0 = video (input 0)   1 = fire   2 = nebula swirl   3 = alignment test pattern
//
// Output is premultiplied alpha, circle-masked.

uniform vec3 uA;
uniform vec4 uB;      // .w = video zoom (1 = cover-fit, >1 = zoom in)
uniform vec3 uC;
uniform vec3 uD;

out vec4 fragColor;

// ---------------------------------------------------------------- noise
// sin-free hash (stays clean at large coordinates on the GPU)
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}

float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; }
    return s;
}

vec3 hueShift(vec3 c, float h) {
    const vec3 k = vec3(0.57735);
    float a = h * 6.2831853;
    float ca = cos(a);
    return c * ca + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - ca);
}

// ---------------------------------------------------------------- scenes
// q: upright coords, -1..1 across the head, +y = world up
vec3 sceneVideo(vec2 q) {
    vec2 res = uTD2DInfos[0].res.zw;
    float aspect = res.x / max(res.y, 1.0);
    vec2 uv = q * 0.5 / max(uB.w, 0.05);
    if (aspect > 1.0) uv.x /= aspect; else uv.y *= aspect;   // cover-fit
    vec2 st = uv + 0.5;
    vec4 c = texture(sTD2DInputs[0], st);
    // outside the image (zoomed out) or transparent pixels -> black (no light)
    float inside = step(0.0, st.x) * step(st.x, 1.0) * step(0.0, st.y) * step(st.y, 1.0);
    return c.rgb * c.a * inside;
}

vec3 sceneFire(vec2 q, float t) {
    float audio = uB.y;
    // velocity seen upright, flame bends away from the direction of motion
    vec2 up = uD.yz;
    vec2 rt = vec2(up.y, -up.x);
    vec2 vel = vec2(dot(uC.xy, rt), dot(uC.xy, up));
    float lean = uC.z;
    float h = 1.0 + 0.6 * audio - 0.25 * clamp(vel.y * lean, -1.0, 1.0);
    float y = (q.y + 0.85) / (1.75 * h);               // 0 at base, 1 at tip
    float yc = clamp(y, 0.0, 1.0);
    float bend = -clamp(vel.x * lean, -1.5, 1.5) * yc * yc * 0.9;
    float x = q.x - bend;

    float n1 = fbm(vec2(x * 2.2, q.y * 1.8 - t * 2.0));
    float n2 = fbm(vec2(x * 4.5 + 3.1, q.y * 3.6 - t * 3.4));
    float width = mix(0.75, 0.04, pow(yc, 0.8)) * (0.9 + 0.3 * audio);
    float d = abs(x + (n1 - 0.5) * 0.7 * yc) / width;
    float f = (1.0 - d) * (1.0 - yc * 0.85);
    f += (n2 - 0.5) * 0.55 * yc;
    f *= smoothstep(-0.05, 0.12, y);
    f = clamp(f * 1.7, 0.0, 1.0);

    vec3 c = vec3(0.0);
    c = mix(c, vec3(0.55, 0.05, 0.01), smoothstep(0.0, 0.25, f));
    c = mix(c, vec3(1.0, 0.35, 0.02), smoothstep(0.2, 0.55, f));
    c = mix(c, vec3(1.0, 0.8, 0.25), smoothstep(0.5, 0.85, f));
    c = mix(c, vec3(1.0, 0.97, 0.85), smoothstep(0.85, 1.0, f));
    // embers: small round sparks drifting up
    vec2 eg = vec2(q.x * 14.0, q.y * 14.0 - t * 5.0);
    vec2 ec = floor(eg);
    vec2 ef = fract(eg) - 0.5 - 0.3 * (vec2(hash(ec + 7.0), hash(ec + 3.0)) - 0.5);
    float e = step(0.93, hash(ec)) * (1.0 - smoothstep(0.05, 0.12, length(ef)));
    e *= smoothstep(0.95, 0.3, length(q)) * smoothstep(-0.3, 0.4, q.y);
    c += vec3(1.0, 0.55, 0.15) * e * (0.6 + audio);
    return c;
}

vec3 sceneNebula(vec2 q, float t) {
    float r = length(q);
    float a = atan(q.y, q.x);
    float audio = uB.y;
    vec2 w = vec2(fbm(q * 2.0 + t * 0.15), fbm(q * 2.0 - t * 0.12 + 5.0));
    float sw = a + 1.5 * r - t * 0.3 + 2.0 * (w.x - 0.5);
    float petals = 0.5 + 0.5 * sin(6.0 * sw + 4.0 * fbm(q * 3.0 + t * 0.2));
    petals = mix(0.5, petals, smoothstep(0.03, 0.35, r));   // no pinwheel spokes at the very center
    float rings = 0.5 + 0.5 * sin(18.0 * r - t * 2.0 - audio * 6.0);
    vec3 c1 = vec3(0.10, 0.25, 0.9), c2 = vec3(0.95, 0.2, 0.6), c3 = vec3(0.2, 0.95, 0.8);
    vec3 c = mix(c1, c2, petals);
    c = mix(c, c3, rings * 0.35);
    c *= 0.35 + 0.9 * fbm(q * 3.0 + w * 2.0 - t * 0.1);
    c += (1.0 - smoothstep(0.0, 0.35, r)) * (0.3 + audio) * vec3(1.0, 0.9, 0.7);
    return c;
}

vec3 sceneTest(vec2 q, vec2 p, float t) {
    // p = raw head-local coords (no upright rotation) so you can see the tracker frame
    float r = length(p);
    vec3 c = vec3(0.05);
    // quadrants: +x red, +y green (head-local)
    c += (p.x > 0.0 ? vec3(0.25, 0.0, 0.0) : vec3(0.0)) + (p.y > 0.0 ? vec3(0.0, 0.25, 0.0) : vec3(0.0));
    float lw = 0.015;
    for (int i = 1; i <= 4; i++) {
        float rr = float(i) * 0.25 - (i == 4 ? 0.02 : 0.0);
        c = mix(c, vec3(1.0), 1.0 - smoothstep(lw, lw * 2.0, abs(r - rr)));
    }
    c = mix(c, vec3(1.0), 1.0 - smoothstep(lw, lw * 2.0, min(abs(p.x), abs(p.y))));
    // world-up arrow (cyan) - should always point at the ceiling
    vec2 a = q - vec2(0.0, 0.55);
    float head = max(abs(a.x) * 2.2 + a.y - 0.22, -a.y);                 // arrow head
    float shaft = max(abs(a.x) - 0.035, max(a.y, -a.y - 0.35));           // arrow shaft
    c = mix(c, vec3(0.0, 1.0, 1.0), 1.0 - smoothstep(0.0, 0.01, min(head, shaft)));
    // blink the center so latency is easy to judge
    c = mix(c, vec3(1.0, 1.0, 0.0), (1.0 - smoothstep(0.04, 0.06, r)) * step(0.5, fract(t)));
    return c;
}

// Time-driven noise loses precision as t grows, so run two copies on wrapped
// clocks offset by half a period and crossfade - no visible pop, ever.
const float PERIOD = 240.0;

vec3 sceneWrapped(int i, vec2 q, float t) {
    float ph = mod(t, PERIOD) / PERIOD;
    float tA = mod(t, PERIOD);
    float tB = mod(t + 0.5 * PERIOD, PERIOD) + 37.0;
    float wA = sin(3.14159265 * ph);
    wA *= wA;
    vec3 a = (i == 1) ? sceneFire(q, tA) : sceneNebula(q, tA);
    vec3 b = (i == 1) ? sceneFire(q, tB) : sceneNebula(q, tB);
    return mix(b, a, wA);
}

vec3 scene(int i, vec2 q, vec2 p, float t) {
    if (i <= 0) return sceneVideo(q);
    if (i <= 2) return sceneWrapped(i, q, t);
    return sceneTest(q, p, fract(t));
}

void main() {
    float t = uA.x;
    vec2 p = vUV.st * 2.0 - 1.0;                    // head-local, -1..1
    vec2 up = normalize(uD.yz + vec2(1e-5));
    vec2 rt = vec2(up.y, -up.x);
    vec2 q = vec2(dot(p, rt), dot(p, up));          // upright

    float s = clamp(uA.y, 0.0, 3.0);
    int s0 = int(floor(s));
    int s1 = min(s0 + 1, 3);
    float m = s - float(s0);
    vec3 c = scene(s0, q, p, t);
    if (m > 0.001) c = mix(c, scene(s1, q, p, t), m);

    c = hueShift(c, uB.x) * uA.z;
    float r = length(p);
    float feather = max(uB.z, 0.002);
    float alpha = (1.0 - smoothstep(1.0 - feather, 1.0, r)) * clamp(uD.x, 0.0, 1.0);
    fragColor = TDOutputSwizzle(vec4(c * alpha, alpha));
}
