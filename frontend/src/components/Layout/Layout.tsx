import { Outlet } from "react-router-dom";
import Header from "./Header";

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="bg-udemy-dark text-gray-400 text-center py-6 text-sm">
        <p>Ace Your Interview &mdash; Practice the skills that get offers</p>
      </footer>
    </div>
  );
}
