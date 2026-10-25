import type { SlideDef } from '../deck/SlideFrame'
import { build } from '../components/layouts'
import team from '../data/team.json'

const initials = (n: string) => n.split(/\s+/).map(w => w[0]).join('').slice(0, 2).toUpperCase()

export default {
  title: "What's next",
  section: "What's next",
  chrome: false,
  steps: 1,
  Component: () => (
    <div className="frame next">
      <div className="next-top">
        <div {...build(0, 'fade')}>
          <span className="kicker">Next</span>
          <ul className="next-list">
            <li><b>Patient-specific</b> heart models</li>
            <li><b>Real-time</b> hazard maps</li>
          </ul>
        </div>
        <div className="team" {...build(0, 'fade', 0.3)}>
          <span className="kicker">Team</span>
          <div className="members">
            {team.members.map(m => (
              <figure key={m.name} className="member">
                {m.photo ? <img src={`${import.meta.env.BASE_URL}${m.photo}`} alt={m.name} /> : <span className="avatar">{initials(m.name)}</span>}
                <figcaption>{m.name}{m.role && <small>{m.role}</small>}</figcaption>
              </figure>
            ))}
          </div>
        </div>
      </div>
      <p className="closing" {...build(1, 'words')}>Simula turns hours of simulation into decisions in seconds.</p>
    </div>
  ),
  notes: `(3:50–4:00)

Next: patient-specific heart models, real-time hazard maps.
Team: (names)
(click) Closing line: "Simula turns hours of simulation into decisions in seconds."`,
} satisfies SlideDef
