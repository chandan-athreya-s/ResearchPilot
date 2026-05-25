import { AnimatePresence, motion } from "framer-motion";
import { Routes, Route, useLocation } from "react-router-dom";
import AppRoutes from "./routes/AppRoutes";

function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen">
      <AnimatePresence mode="wait">
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -14 }}
          transition={{ duration: 0.35 }}
          className="px-4 py-4 sm:px-6 lg:px-8"
        >
          <Routes location={location}>
            <Route path="/*" element={<AppRoutes />} />
          </Routes>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default App;
