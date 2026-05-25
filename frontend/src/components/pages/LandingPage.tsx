import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card } from "../ui/card";

const features = [
  "AI-powered multi-agent research orchestration",
  "Live workspace with query, retrieval and report views",
  "Paper cards with relevance, sources, and PDF access",
  "Agent execution timeline for visibility and trust",
  "Export reports and reference summaries to PDF",
];

export default function LandingPage() {
  return (
    <div className="mx-auto max-w-7xl">
      <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] xl:gap-16">
        <section className="space-y-8 pt-12 pb-16">
          <Badge variant="success" className="uppercase tracking-[0.24em] text-[11px]">
            ResearchPilot
          </Badge>
          <div className="space-y-4">
            <motion.h1
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45 }}
              className="max-w-3xl text-5xl font-semibold tracking-tight text-white sm:text-6xl"
            >
              Intelligent research assistance for academic discovery.
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45, delay: 0.1 }}
              className="max-w-2xl text-lg leading-8 text-slate-300"
            >
              ResearchPilot unifies multi-agent retrieval, evidence extraction, and report generation into a polished AI workspace.
              Submit queries, trace agent progress, review paper sources, and generate production-ready summaries.
            </motion.p>
          </div>

          <div className="flex flex-col gap-4 sm:flex-row">
            <Link to="/workspace">
              <Button className="w-full sm:w-auto">Launch Workspace</Button>
            </Link>
            <a href="https://github.com/" className="w-full sm:w-auto">
              <Button variant="secondary">View Documentation</Button>
            </a>
          </div>
        </section>

        <section className="space-y-6 pt-16 lg:pt-24">
          <Card className="space-y-6">
            <div className="space-y-3">
              <p className="text-sm uppercase tracking-[0.25em] text-accent-400">Workspace preview</p>
              <h2 className="text-2xl font-semibold text-white">Built for rapid academic workflows</h2>
            </div>
            <div className="space-y-4">
              {features.map((feature) => (
                <div key={feature} className="rounded-3xl border border-white/10 bg-surface-800/80 p-4 transition hover:border-accent-500/30">
                  <p className="text-sm text-slate-200">{feature}</p>
                </div>
              ))}
            </div>
          </Card>
        </section>
      </div>
    </div>
  );
}
