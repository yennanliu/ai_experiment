import { NotesProvider } from './contexts/NotesContext';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { NotesList } from './components/notes/NotesList';
import { NoteEditor } from './components/notes/NoteEditor';

function App() {
  return (
    <NotesProvider>
      <div className="flex flex-col h-screen">
        <Header />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <div className="flex-1 flex overflow-hidden">
            <div className="w-80 border-r border-gray-200 overflow-hidden">
              <NotesList />
            </div>
            <div className="flex-1 bg-white overflow-hidden">
              <NoteEditor />
            </div>
          </div>
        </div>
      </div>
    </NotesProvider>
  );
}

export default App;
