// glsl_calib_pixel.glsl - projector-space calibration target
// uA = (target u px, target v px, hold progress 0..1)
// uB = (active 0/1, time s, unused)

uniform vec3 uA;
uniform vec3 uB;

out vec4 fragColor;

void main() {
    vec2 res = uTDOutputInfo.res.zw;
    vec2 px = vUV.st * res;
    vec3 c = vec3(0.0);

    // frame around the projector image so you can see its full extent
    float edge = min(min(px.x, res.x - px.x), min(px.y, res.y - px.y));
    c += vec3(0.25) * (1.0 - step(3.0, edge));

    if (uB.x > 0.5) {
        vec2 d = px - uA.xy;
        float r = length(d);
        // crosshair
        float cross = (abs(d.x) < 1.5 && abs(d.y) < 70.0) || (abs(d.y) < 1.5 && abs(d.x) < 70.0) ? 1.0 : 0.0;
        cross *= step(6.0, r);
        // bright center dot so the reflective marker lights up
        float dot_ = 1.0 - smoothstep(4.0, 6.0, r);
        // progress ring
        float ring = 1.0 - smoothstep(2.0, 3.5, abs(r - 34.0));
        float ang = fract(atan(d.x, d.y) / 6.2831853 + 1.0);
        float done = step(ang, uA.z);
        vec3 ringc = mix(vec3(0.35), vec3(0.1, 1.0, 0.3), done);
        float pulse = 0.6 + 0.4 * sin(uB.y * 6.0);
        c = max(c, vec3(1.0) * cross);
        c = max(c, vec3(1.0) * dot_);
        c = max(c, ringc * ring * (uA.z > 0.0 ? 1.0 : pulse));
    }
    fragColor = TDOutputSwizzle(vec4(c, 1.0));
}
