import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'CS676 Algorithms for Data Science',
  tagline: 'Pace University — lecture notes, notebooks, and capstone projects',
  favicon: 'img/favicon.svg',

  url: 'https://yiqiao-yin.github.io',
  baseUrl: '/pace-u-cs676/',
  organizationName: 'yiqiao-yin',
  projectName: 'pace-u-cs676',
  trailingSlash: false,

  // Existing notes contain cross-file links written for GitHub's renderer;
  // warn rather than fail the production build on those.
  onBrokenLinks: 'warn',
  onBrokenAnchors: 'warn',

  i18n: {defaultLocale: 'en', locales: ['en']},

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          // The slide PDF lives under docs/ but is not a page.
          exclude: ['slide_doc/**'],
          editUrl:
            'https://github.com/yiqiao-yin/pace-u-cs676/edit/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  // KaTeX styles for the inline/display math in the lecture notes.
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
    mermaid: {
      theme: {light: 'dark', dark: 'dark'},
    },
    navbar: {
      title: 'CS676',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'courseSidebar',
          position: 'left',
          label: 'Course Notes',
        },
        {
          to: '/docs/capstone',
          label: 'Capstone',
          position: 'left',
        },
        {
          href: 'https://github.com/yiqiao-yin/pace-u-cs676',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Course',
          items: [
            {label: 'Introduction', to: '/docs/introduction'},
            {label: 'Capstone Projects', to: '/docs/capstone'},
            {label: 'Presentation Guidance', to: '/docs/final_guidance'},
          ],
        },
        {
          title: 'Resources',
          items: [
            {label: 'Introduction to Statistical Learning', href: 'https://www.statlearning.com/'},
            {label: 'Notebooks', href: 'https://github.com/yiqiao-yin/pace-u-cs676/tree/main/notebooks'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Yiqiao Yin — CS676, Pace University.`,
    },
    prism: {
      theme: prismThemes.vsDark,
      darkTheme: prismThemes.vsDark,
      additionalLanguages: ['python', 'r', 'bash', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
