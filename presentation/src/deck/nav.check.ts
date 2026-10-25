// Self-check for the navigation math: `npm run check` (node strips the types).
import assert from 'node:assert/strict'
import { advance, format, parse, progress } from './nav.ts'

const steps = [0, 2, 1] // slide 1 has no builds, slide 2 has two, slide 3 one
const walk = [{ i: 0, s: 0 }, { i: 1, s: 0 }, { i: 1, s: 1 }, { i: 1, s: 2 }, { i: 2, s: 0 }, { i: 2, s: 1 }]
for (let k = 0; k < walk.length - 1; k++) {
  assert.deepEqual(advance(steps, walk[k], 1), walk[k + 1])
  assert.deepEqual(advance(steps, walk[k + 1], -1), walk[k]) // back lands on the previous slide's last step
}
assert.deepEqual(advance(steps, walk[0], -1), walk[0])
assert.deepEqual(advance(steps, walk[5], 1), walk[5])

assert.deepEqual(parse('#/2/1', steps), { i: 1, s: 1 })
assert.deepEqual(parse('#/2/9', steps), { i: 1, s: 2 }) // step clamped to the slide
assert.deepEqual(parse('#/99', steps), { i: 2, s: 0 })
assert.deepEqual(parse('', steps), { i: 0, s: 0 })
for (const p of walk) assert.deepEqual(parse(format(p), steps), p)

assert.equal(progress(steps, walk[0]), 0)
assert.equal(progress(steps, walk[5]), 1)
assert.equal(progress(steps, walk[3]), 3 / 5)
console.log('nav ok')
