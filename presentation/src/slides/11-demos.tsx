import type { SlideDef } from '../deck/SlideFrame'
import { Frame } from '../components/layouts'

export default {
  title: 'Live: tsunami and cardiac spiral waves',
  tone: 'ember',
  section: 'Demo',
  Component: () => <Frame title="Live: tsunami and cardiac spiral waves" lede="Rendered live in three.js from model rollouts." />,
  notes: `Live embeds of viz/tsunami and viz/heart.`,
} satisfies SlideDef
