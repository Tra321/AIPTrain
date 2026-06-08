import { useEffect, useMemo, useState } from 'react'
import './App.css'

const STORAGE_KEY = 'todo-reminder-items'

const defaultTodos = [
  {
    id: crypto.randomUUID(),
    title: '整理今天的重点任务',
    note: '把最需要准时完成的事情放在顶部。',
    dueAt: getLocalInputValue(45),
    priority: 'high',
    done: false,
    notified: false,
    createdAt: Date.now(),
  },
  {
    id: crypto.randomUUID(),
    title: '复盘已完成清单',
    note: '完成后勾选，页面会自动统计进度。',
    dueAt: getLocalInputValue(180),
    priority: 'medium',
    done: false,
    notified: false,
    createdAt: Date.now() - 1000,
  },
]

const priorityMap = {
  high: { label: '高', rank: 0 },
  medium: { label: '中', rank: 1 },
  low: { label: '低', rank: 2 },
}

function getLocalInputValue(offsetMinutes = 30) {
  const date = new Date(Date.now() + offsetMinutes * 60 * 1000)
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000)
  return local.toISOString().slice(0, 16)
}

function readStoredTodos() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    return saved ? JSON.parse(saved) : defaultTodos
  } catch {
    return defaultTodos
  }
}

function formatDueTime(value) {
  if (!value) return '未设置'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function getStatus(todo) {
  if (todo.done) return 'done'
  if (!todo.dueAt) return 'open'

  const dueTime = new Date(todo.dueAt).getTime()
  const diff = dueTime - Date.now()

  if (diff <= 0) return 'overdue'
  if (diff <= 30 * 60 * 1000) return 'soon'
  return 'open'
}

function App() {
  const [todos, setTodos] = useState(readStoredTodos)
  const [filter, setFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [form, setForm] = useState({
    title: '',
    note: '',
    dueAt: getLocalInputValue(60),
    priority: 'medium',
  })
  const [permission, setPermission] = useState(
    'Notification' in window ? Notification.permission : 'unsupported',
  )

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(todos))
  }, [todos])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setTodos((current) =>
        current.map((todo) => {
          const shouldNotify =
            !todo.done &&
            !todo.notified &&
            todo.dueAt &&
            new Date(todo.dueAt).getTime() <= Date.now()

          if (!shouldNotify) return todo

          if ('Notification' in window && Notification.permission === 'granted') {
            new Notification('待办事项提醒', {
              body: todo.title,
              tag: todo.id,
            })
          }

          return { ...todo, notified: true }
        }),
      )
    }, 15000)

    return () => window.clearInterval(timer)
  }, [])

  const stats = useMemo(() => {
    const total = todos.length
    const done = todos.filter((todo) => todo.done).length
    const overdue = todos.filter((todo) => getStatus(todo) === 'overdue').length
    const soon = todos.filter((todo) => getStatus(todo) === 'soon').length

    return { total, done, overdue, soon }
  }, [todos])

  const visibleTodos = useMemo(() => {
    const keyword = query.trim().toLowerCase()

    return todos
      .filter((todo) => {
        const status = getStatus(todo)

        if (filter === 'active' && todo.done) return false
        if (filter === 'done' && !todo.done) return false
        if (filter === 'overdue' && status !== 'overdue') return false
        if (!keyword) return true

        return `${todo.title} ${todo.note}`.toLowerCase().includes(keyword)
      })
      .sort((a, b) => {
        if (a.done !== b.done) return Number(a.done) - Number(b.done)
        const aDue = a.dueAt ? new Date(a.dueAt).getTime() : Infinity
        const bDue = b.dueAt ? new Date(b.dueAt).getTime() : Infinity
        if (aDue !== bDue) return aDue - bDue
        return priorityMap[a.priority].rank - priorityMap[b.priority].rank
      })
  }, [filter, query, todos])

  function handleSubmit(event) {
    event.preventDefault()
    const title = form.title.trim()

    if (!title) return

    setTodos((current) => [
      {
        id: crypto.randomUUID(),
        title,
        note: form.note.trim(),
        dueAt: form.dueAt,
        priority: form.priority,
        done: false,
        notified: false,
        createdAt: Date.now(),
      },
      ...current,
    ])

    setForm({
      title: '',
      note: '',
      dueAt: getLocalInputValue(60),
      priority: 'medium',
    })
  }

  function toggleTodo(id) {
    setTodos((current) =>
      current.map((todo) =>
        todo.id === id ? { ...todo, done: !todo.done } : todo,
      ),
    )
  }

  function removeTodo(id) {
    setTodos((current) => current.filter((todo) => todo.id !== id))
  }

  function snoozeTodo(id, minutes) {
    setTodos((current) =>
      current.map((todo) =>
        todo.id === id
          ? { ...todo, dueAt: getLocalInputValue(minutes), notified: false }
          : todo,
      ),
    )
  }

  async function requestNotificationPermission() {
    if (!('Notification' in window)) return
    const result = await Notification.requestPermission()
    setPermission(result)
  }

  return (
    <main className="app-shell">
      <section className="summary-band" aria-label="待办提醒概览">
        <div>
          <p className="eyebrow">Todo Reminder</p>
          <h1>待办事项提醒工具</h1>
          <p className="intro">
            记录任务、设置提醒时间，并在浏览器中接收准点提醒。
          </p>
        </div>

        <div className="stats-grid">
          <div className="stat">
            <span>{stats.total}</span>
            <small>全部</small>
          </div>
          <div className="stat">
            <span>{stats.soon}</span>
            <small>即将到期</small>
          </div>
          <div className="stat danger">
            <span>{stats.overdue}</span>
            <small>已逾期</small>
          </div>
          <div className="stat success">
            <span>{stats.done}</span>
            <small>已完成</small>
          </div>
        </div>
      </section>

      <section className="workspace">
        <form className="todo-form" onSubmit={handleSubmit}>
          <div className="section-title">
            <h2>新建提醒</h2>
            <button
              className="ghost-button"
              type="button"
              onClick={requestNotificationPermission}
              disabled={permission === 'granted' || permission === 'unsupported'}
            >
              {permission === 'granted'
                ? '通知已开启'
                : permission === 'unsupported'
                  ? '不支持通知'
                  : '开启通知'}
            </button>
          </div>

          <label>
            任务名称
            <input
              value={form.title}
              onChange={(event) =>
                setForm((current) => ({ ...current, title: event.target.value }))
              }
              placeholder="例如：提交周报"
              maxLength={60}
            />
          </label>

          <label>
            备注
            <textarea
              value={form.note}
              onChange={(event) =>
                setForm((current) => ({ ...current, note: event.target.value }))
              }
              placeholder="补充地点、材料或注意事项"
              rows="4"
              maxLength={160}
            />
          </label>

          <div className="form-grid">
            <label>
              提醒时间
              <input
                type="datetime-local"
                value={form.dueAt}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    dueAt: event.target.value,
                  }))
                }
              />
            </label>

            <label>
              优先级
              <select
                value={form.priority}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    priority: event.target.value,
                  }))
                }
              >
                <option value="high">高优先级</option>
                <option value="medium">中优先级</option>
                <option value="low">低优先级</option>
              </select>
            </label>
          </div>

          <button className="primary-button" type="submit">
            添加待办
          </button>
        </form>

        <section className="todo-panel">
          <div className="toolbar">
            <div className="segmented" aria-label="筛选待办">
              {[
                ['all', '全部'],
                ['active', '进行中'],
                ['overdue', '已逾期'],
                ['done', '已完成'],
              ].map(([value, label]) => (
                <button
                  className={filter === value ? 'active' : ''}
                  key={value}
                  type="button"
                  onClick={() => setFilter(value)}
                >
                  {label}
                </button>
              ))}
            </div>

            <input
              className="search-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索任务"
            />
          </div>

          <div className="todo-list" aria-live="polite">
            {visibleTodos.length === 0 ? (
              <div className="empty-state">
                <h2>没有匹配的待办</h2>
                <p>换个筛选条件，或者添加一条新的提醒。</p>
              </div>
            ) : (
              visibleTodos.map((todo) => {
                const status = getStatus(todo)

                return (
                  <article className={`todo-item ${status}`} key={todo.id}>
                    <button
                      className="check-button"
                      type="button"
                      aria-label={todo.done ? '标记为未完成' : '标记为完成'}
                      onClick={() => toggleTodo(todo.id)}
                    >
                      {todo.done ? '✓' : ''}
                    </button>

                    <div className="todo-content">
                      <div className="todo-heading">
                        <h3>{todo.title}</h3>
                        <span className={`priority ${todo.priority}`}>
                          {priorityMap[todo.priority].label}
                        </span>
                      </div>
                      {todo.note ? <p>{todo.note}</p> : null}
                      <div className="meta-row">
                        <span>{formatDueTime(todo.dueAt)}</span>
                        <span>
                          {status === 'overdue'
                            ? '已到提醒时间'
                            : status === 'soon'
                              ? '30 分钟内到期'
                              : todo.done
                                ? '任务已完成'
                                : '等待提醒'}
                        </span>
                      </div>
                    </div>

                    <div className="item-actions">
                      {!todo.done ? (
                        <button type="button" onClick={() => snoozeTodo(todo.id, 10)}>
                          延后10分钟
                        </button>
                      ) : null}
                      <button type="button" onClick={() => removeTodo(todo.id)}>
                        删除
                      </button>
                    </div>
                  </article>
                )
              })
            )}
          </div>
        </section>
      </section>
    </main>
  )
}

export default App
