import { Route, Routes } from "react-router-dom";
import LandingPage from "../components/pages/LandingPage";
import WorkspacePage from "../components/pages/WorkspacePage";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/workspace" element={<WorkspacePage />} />
    </Routes>
  );
}
