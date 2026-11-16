// glsl_out_pixel.glsl - projector-space afterglow trail + nudge
// input 0 = render_projector, input 1 = feedback (previous output)
// uA = (afterglow decay 0..0.98, rise px/frame, sway px)
// uB = (nudge x px, nudge y px, time s)

uniform vec3 uA;
uniform vec3 uB;

out vec4 fragColor;

void main() {
    vec2 res = uTD2DInfos[0].res.zw;
    vec4 cur = texture(sTD2DInputs[0], vUV.st - uB.xy / res);
    vec4 outc = cur;
    if (uA.x > 0.001) {
        float sway = sin(vUV.t * 23.0 + uB.z * 3.0) * uA.z;
        vec2 off = vec2(sway, uA.y) / res;
        vec4 prev = texture(sTD2DInputs[1], vUV.st - off);
        // cool the trail slightly toward red as it fades
        prev.rgb *= vec3(1.0, 0.92, 0.8);
        outc = max(cur, prev * uA.x);
    }
    fragColor = TDOutputSwizzle(outc);
}
