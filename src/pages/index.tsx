import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';

type Session = {
  num: string;
  title: string;
  blurb: string;
  to: string;
};

const SESSIONS: Session[] = [
  {num: '01', title: 'Introduction', blurb: 'Course overview and Python setup.', to: '/docs/introduction'},
  {num: '02', title: 'Basics in Statistical Learning', blurb: 'Core concepts and definitions.', to: '/docs/basics_in_stat_learning'},
  {num: '03', title: 'Linear Regression', blurb: 'Simple, multiple, and model accuracy.', to: '/docs/linear_regression'},
  {num: '04', title: 'Classification', blurb: 'Logistic regression, LDA, metrics.', to: '/docs/classification'},
  {num: '05', title: 'Sampling and Bootstrap', blurb: 'Resampling and cross validation.', to: '/docs/sampling_and_bootstrap'},
  {num: '06', title: 'Model Selection', blurb: 'Ridge, lasso, regularization.', to: '/docs/model_selection'},
  {num: '07', title: 'Beyond Linearity', blurb: 'Polynomials, step functions, splines.', to: '/docs/going_beyond_linearity'},
  {num: '08', title: 'Tree-Based Methods', blurb: 'Trees, random forests, boosting.', to: '/docs/tree_based_model'},
  {num: '09', title: 'Support Vector Machine', blurb: 'SVM for classification and regression.', to: '/docs/support_vector_machine'},
  {num: '10', title: 'Deep Learning', blurb: 'Neural networks and architectures.', to: '/docs/neural_networks'},
  {num: '11', title: 'Unsupervised Metrics', blurb: 'Clustering and evaluation.', to: '/docs/unsupervised'},
  {num: '12', title: 'Capstone Projects', blurb: 'Project specs, deadlines, rubrics.', to: '/docs/capstone'},
];

function Hero() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <p className={styles.eyebrow}>Pace University</p>
        <h1 className={styles.heroTitle}>{siteConfig.title}</h1>
        <p className={styles.heroSubtitle}>
          Essential algorithms for data analytics, with a computational emphasis —
          from linear models through deep learning, ending in a full-stack
          LLM capstone.
        </p>
        <div className={styles.heroButtons}>
          <Link className={styles.primaryBtn} to="/docs/introduction">
            Start the course
          </Link>
          <Link className={styles.secondaryBtn} to="/docs/capstone">
            Capstone projects
          </Link>
        </div>
      </div>
    </header>
  );
}

function SessionGrid() {
  return (
    <section className={styles.gridSection}>
      <h2 className={styles.sectionTitle}>Sessions</h2>
      <div className={styles.grid}>
        {SESSIONS.map((s) => (
          <Link key={s.num} to={s.to} className={styles.card}>
            <span className={styles.cardNum}>{s.num}</span>
            <span className={styles.cardTitle}>{s.title}</span>
            <span className={styles.cardBlurb}>{s.blurb}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Home"
      description="CS676 Algorithms for Data Science — Pace University course notes.">
      <Hero />
      <main>
        <SessionGrid />
      </main>
    </Layout>
  );
}
